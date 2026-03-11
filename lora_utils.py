"""
LoRA utilities for MetalThroat v2.

Provides a minimal, dependency-free LoRA implementation that works with
audiocraft's MusicGen without wrapping or replacing the LMModel class.

Why manual LoRA (not PEFT get_peft_model)?
  PEFT wraps model.lm in a PeftModel that lacks compute_predictions().
  The entire training/inference loop depends on model.lm.compute_predictions().
  Manual LoRA keeps the LMModel interface intact.

Target modules (nn.Linear only — MusicGen's in_proj_weight is a fused raw
Parameter, not an nn.Linear, so it cannot be targeted by module scanners):
  - out_proj   : attention output projection (self_attn + cross_attention)
  - linear1    : FFN up-projection   [1024 -> 4096]
  - linear2    : FFN down-projection [4096 -> 1024]

At r=128, alpha=256, dropout=0.05 across 24 transformer layers this yields
~44M trainable parameters out of ~420M total — matching the ~40M sweet spot
from "Exploring Adapter Design Tradeoffs for Low Resource Music Generation"
(arXiv:2506.21298, ACM MM 2025). Dropout is motivated by arXiv:2404.09610
which shows LoRA Dropout outperforms vanilla LoRA on small datasets.

rsLoRA note:
  Setting use_rslora=True changes scaling from alpha/r to alpha/sqrt(r).
  At r=128 this increases scaling ~11x (2.0 → 22.6), which requires reducing
  LEARNING_RATE to ~1e-4. Not the default — use for experimentation only.
  See: arXiv:2312.03732
"""
import math
import torch
import torch.nn as nn
from pathlib import Path


# ── Default LoRA hyperparameters ─────────────────────────────────────────────
LORA_R        = 128
LORA_ALPHA    = 256
LORA_DROPOUT  = 0.05   # arXiv:2404.09610 — helps generalization on small datasets
LORA_TARGETS  = {"out_proj", "linear1", "linear2"}

# rsLoRA experiment flag (see module docstring before enabling)
# USE_RSLORA  = False   # change alpha/r → alpha/sqrt(r) if True; reduce LR to 1e-4


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank adapter + optional dropout.

    The base weight is frozen at init. Only lora_A and lora_B are trainable.
    At init, lora_B is zero so the adapter starts as an identity delta (no
    change to the base model's output on step 0).

    Dropout is applied between lora_A and lora_B (arXiv:2404.09610). At p=0.05
    this provides sparsity regularization without destabilizing training.
    """
    def __init__(self, base: nn.Linear, r: int, alpha: int,
                 dropout: float = 0.0, use_rslora: bool = False):
        super().__init__()
        self.base         = base
        self.lora_A       = nn.Linear(base.in_features,  r, bias=False)
        self.lora_B       = nn.Linear(r, base.out_features, bias=False)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        # rsLoRA uses alpha/sqrt(r) for gradient stability at high ranks
        self.scaling      = (alpha / math.sqrt(r)) if use_rslora else (alpha / r)

        # Standard LoRA init: A ~ kaiming, B = 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Move adapters to the same device and dtype as the base layer
        device = base.weight.device
        dtype  = base.weight.dtype
        self.lora_A = self.lora_A.to(device=device, dtype=dtype)
        self.lora_B = self.lora_B.to(device=device, dtype=dtype)

        # Freeze the base layer
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_dropout(self.lora_A(x)))


def inject_lora(
    module: nn.Module,
    target_names: set,
    r: int,
    alpha: int,
    dropout: float = 0.0,
    use_rslora: bool = False,
) -> None:
    """Recursively replace nn.Linear submodules whose name is in target_names.

    Operates in-place. After this call:
      - Matching nn.Linear instances become LoRALinear (base frozen, adapters trainable)
      - All other parameters remain unchanged (requires_grad not touched here;
        call freeze_base_model() first to freeze everything before injecting)
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, r, alpha,
                                             dropout=dropout,
                                             use_rslora=use_rslora))
        else:
            inject_lora(child, target_names, r, alpha,
                        dropout=dropout, use_rslora=use_rslora)


def freeze_base_model(model) -> None:
    """Freeze all parameters in model.lm and model.compression_model.

    Call this BEFORE inject_lora(). inject_lora() will then unfreeze only
    the LoRA adapter weights (lora_A.weight, lora_B.weight).
    """
    for p in model.lm.parameters():
        p.requires_grad_(False)
    for p in model.compression_model.parameters():
        p.requires_grad_(False)


def count_trainable(model) -> tuple[int, int]:
    """Print and return (trainable_params, total_lm_params)."""
    trainable = sum(p.numel() for p in model.lm.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.lm.parameters())
    print(f"Trainable LM params: {trainable:,} / {total:,}  ({100 * trainable / total:.1f}%)")
    return trainable, total


def get_lora_state_dict(model) -> dict:
    """Return only the trainable (LoRA) parameters as a state dict."""
    return {
        name: param.data.clone()
        for name, param in model.lm.named_parameters()
        if param.requires_grad
    }


def load_lora_checkpoint(model, path: Path | str, device: str) -> dict:
    """Load LoRA adapter weights from a checkpoint into an already-injected model.

    The model must already have inject_lora() applied so the LoRA parameter
    names exist. This function does NOT call load_state_dict() on the full
    model (which would fail due to the LoRALinear key name changes).

    Returns the full checkpoint dict (for epoch/loss metadata).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    lm_params = dict(model.lm.named_parameters())
    for name, data in ckpt['lora_state_dict'].items():
        if name in lm_params:
            lm_params[name].data.copy_(data)
        else:
            print(f"[warn] checkpoint key not found in model: {name}")
    return ckpt
