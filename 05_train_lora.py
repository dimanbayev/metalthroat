"""
MetalThroat v2: LoRA Fine-Tuning Script

Trains low-rank adapters on MusicGen-small's attention/FFN output projections.
Only ~44M parameters are updated; the 376M base weights remain frozen.
This dramatically reduces overfitting compared to the full fine-tune in v1.

Improvements over the baseline LoRA design:
  - LoRA Dropout (p=0.05): sparsity regularization for small-dataset generalization
    (arXiv:2404.09610)
  - LR Warmup (1 epoch): linear warmup to 3e-4 before cosine decay, preventing
    early gradient instability on the audio token prediction loss landscape
  - Pitch-Shift Augmentation: ±2 semitones on 30% of training clips, diversifying
    the harmonic distribution without distorting throat-singing timbre

All audiocraft bugs still apply: float32 only, no autocast, nan_to_num before loss.

Usage:
    python 05_train_lora.py

Smoke test (2 epochs):
    Temporarily set TARGET_TOTAL_EPOCHS = 2, run, verify:
      - "Trainable LM params: 44,040,192 / ..." is printed
      - Epoch 1 LR is low (warmup), epoch 2 LR is near 3e-4
      - No NaN losses
      - checkpoints_lora/checkpoint_epoch001.pt is ~176 MB
      - checkpoint contains lora_config.dropout = 0.05
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import torch.nn.functional as F
import torchaudio.functional as TAF
import soundfile as sf
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ConditioningAttributes
import time
from tqdm import tqdm

from lora_utils import (
    inject_lora, freeze_base_model, count_trainable,
    get_lora_state_dict, load_lora_checkpoint,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
)

# ── Config ──────────────────────────────────────────────────────────────────
TARGET_TOTAL_EPOCHS = 60
BATCH_SIZE          = 4
LEARNING_RATE       = 3e-4    # Standard LoRA LR — higher than full fine-tune
WARMUP_EPOCHS       = 1       # Linear warmup before cosine decay
GRAD_CLIP           = 1.0
EARLY_STOP_PATIENCE = 15
DEVICE              = 'cuda'
CHECKPOINT_DIR      = Path('checkpoints_lora')
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Augmentation
AUGMENT_PITCH       = True    # Pitch-shift ±2 semitones on 30% of training clips

# ── Dataset ──────────────────────────────────────────────────────────────────
with open('dataset/dataset_config.json') as f:
    cfg = json.load(f)

DATA_DIR       = Path(cfg['data_dir'])
TRAIN_MANIFEST = Path(cfg['train_manifest'])
VAL_MANIFEST   = Path(cfg['val_manifest'])
SAMPLE_RATE    = cfg['sample_rate']
CLIP_DURATION  = cfg['clip_duration']


class ThroatSingingDataset(Dataset):
    def __init__(self, manifest_path, data_dir, sample_rate=32000, clip_duration=10.0,
                 augment=False):
        self.data_dir    = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * clip_duration)
        self.augment     = augment
        self.records     = []
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        aug_str = " (augment=ON)" if augment else ""
        print(f"Loaded {len(self.records)} records from {Path(manifest_path).name}{aug_str}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record    = self.records[idx]
        audio_np, sr = sf.read(str(self.data_dir / record['path']))
        audio     = torch.from_numpy(audio_np).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.shape[-1] >= self.max_samples:
            audio = audio[:, :self.max_samples]
        else:
            audio = F.pad(audio, (0, self.max_samples - audio.shape[-1]))

        return audio, record['description'], self.augment


def compute_loss(logits, codes, mask):
    B, K, T, card = logits.shape
    # MusicGen uses a codebook delay pattern: codebook k has k undefined (NaN)
    # positions at the start. The mask correctly excludes them, but NaN * 0 = NaN
    # in IEEE 754. Replace NaN with 0 before cross_entropy.
    logits         = logits.nan_to_num(nan=0.0)
    logits_flat    = logits.reshape(-1, card)
    codes_flat     = codes.reshape(-1)
    mask_flat      = mask.reshape(-1)
    loss_per_token = F.cross_entropy(logits_flat, codes_flat, reduction='none')
    masked_loss    = loss_per_token * mask_flat.float()
    return masked_loss.sum() / (mask_flat.sum() + 1e-8)


def gpu_pitch_shift_batch(audio: torch.Tensor, sr: int) -> torch.Tensor:
    """Per-clip GPU pitch shift via resample trick (27x faster than CPU phase vocoder).

    Applies ±1 or ±2 semitone shift to each clip independently with p=0.3.
    Uses torchaudio.functional.resample which is CUDA-accelerated.
    Changes pitch + tempo slightly (acceptable for training augmentation).
    """
    B, C, T = audio.shape
    out = audio.clone()
    for i in range(B):
        if AUGMENT_PITCH and torch.rand(1).item() < 0.3:
            n_steps = int(torch.randint(-2, 3, (1,)).item())
            if n_steps != 0:
                factor  = 2 ** (n_steps / 12.0)
                shifted = TAF.resample(out[i], sr, int(sr * factor))
                if shifted.shape[-1] >= T:
                    out[i] = shifted[:, :T]
                else:
                    out[i] = F.pad(shifted, (0, T - shifted.shape[-1]))
    return out


def train_epoch(model, loader, optimizer, device):
    model.lm.train()
    total_loss   = 0
    num_batches  = 0
    nan_batches  = 0
    trainable_params = [p for p in model.lm.parameters() if p.requires_grad]
    for audio, descs, do_aug in tqdm(loader, desc="train", leave=False, dynamic_ncols=True):
        audio      = audio.to(device)
        # Pitch-shift augmentation on GPU (27x faster than CPU phase vocoder).
        # do_aug[0] is True for train loader, False for val loader.
        if do_aug[0]:
            audio = gpu_pitch_shift_batch(audio, SAMPLE_RATE)
        with torch.no_grad():
            codes, _ = model.compression_model.encode(audio)
        conditions = [ConditioningAttributes(text={'description': d}) for d in descs]
        # Autocast is deliberately DISABLED.
        # audiocraft 1.3.0 has dtype inconsistencies between LM layers; autocast
        # reintroduces float16/bfloat16 mismatches that cause NaN loss.
        out  = model.lm.compute_predictions(codes=codes, conditions=conditions)
        loss = compute_loss(out.logits, codes, out.mask)
        if torch.isnan(loss):
            nan_batches += 1
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
        optimizer.step()
        total_loss  += loss.item()
        num_batches += 1
    if nan_batches:
        print(f"  [warn] {nan_batches} NaN batches skipped")
    return total_loss / num_batches if num_batches > 0 else float('nan')


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.lm.eval()
    total_loss  = 0
    num_batches = 0
    for audio, descs, _ in tqdm(loader, desc="val  ", leave=False, dynamic_ncols=True):
        audio      = audio.to(device)
        codes, _   = model.compression_model.encode(audio)
        conditions = [ConditioningAttributes(text={'description': d}) for d in descs]
        out  = model.lm.compute_predictions(codes=codes, conditions=conditions)
        loss = compute_loss(out.logits, codes, out.mask)
        if not torch.isnan(loss):
            total_loss  += loss.item()
            num_batches += 1
    model.lm.train()
    return total_loss / num_batches if num_batches > 0 else float('nan')


# ── Auto-detect latest LoRA checkpoint ───────────────────────────────────────
all_ckpts = sorted(CHECKPOINT_DIR.glob('checkpoint_epoch*.pt'))
if not all_ckpts:
    print("No LoRA checkpoints found — starting from scratch (pretrained weights)")
    ckpt_path = None
else:
    ckpt_path = all_ckpts[-1]
    print(f"Latest LoRA checkpoint: {ckpt_path.name}")

# ── Datasets & loaders ───────────────────────────────────────────────────────
print("\nLoading datasets...")
train_ds = ThroatSingingDataset(TRAIN_MANIFEST, DATA_DIR, SAMPLE_RATE, CLIP_DURATION,
                                 augment=True)
val_ds   = ThroatSingingDataset(VAL_MANIFEST,   DATA_DIR, SAMPLE_RATE, CLIP_DURATION,
                                 augment=False)  # never augment val
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Load model + inject LoRA ─────────────────────────────────────────────────
print("Loading MusicGen-small...")
model = MusicGen.get_pretrained('facebook/musicgen-small', device=DEVICE)
model.lm = model.lm.float()  # float32 throughout — no autocast

# Step 1: freeze everything
freeze_base_model(model)
model.compression_model.eval()

# Step 2: inject LoRA adapters (unfreezes only lora_A / lora_B weights)
inject_lora(model.lm, LORA_TARGETS, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
count_trainable(model)

# ── Resume from LoRA checkpoint ──────────────────────────────────────────────
if ckpt_path and ckpt_path.exists():
    ckpt        = load_lora_checkpoint(model, ckpt_path, DEVICE)
    start_epoch = ckpt['epoch']
    print(f"Resumed from epoch {start_epoch}  "
          f"train={ckpt['train_loss']:.4f}  val={ckpt['val_loss']:.4f}")
else:
    start_epoch = 0
    print("Starting fresh from pretrained base weights + zero LoRA adapters")

if start_epoch >= TARGET_TOTAL_EPOCHS:
    print(f"\nAlready at epoch {start_epoch} >= {TARGET_TOTAL_EPOCHS}. Nothing to do.")
    print("Run 06_generate_samples_lora.py to create showcase audio.")
    exit(0)

# ── Optimizer & scheduler ─────────────────────────────────────────────────────
trainable_params = [p for p in model.lm.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.0)

# SequentialLR: 1-epoch linear warmup (1e-3 * LR → LR), then cosine decay
warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                         total_iters=WARMUP_EPOCHS)
cosine_sched = CosineAnnealingLR(optimizer,
                                  T_max=max(1, TARGET_TOTAL_EPOCHS - WARMUP_EPOCHS),
                                  eta_min=1e-6)
scheduler = SequentialLR(optimizer,
                          schedulers=[warmup_sched, cosine_sched],
                          milestones=[WARMUP_EPOCHS])

# Fast-forward scheduler to match current epoch (SequentialLR handles milestone)
for _ in range(start_epoch):
    scheduler.step()

best_val   = float('inf')
best_epoch = start_epoch

if ckpt_path and ckpt_path.exists():
    loaded_val = ckpt['val_loss']
    if not (loaded_val != loaded_val):  # not NaN
        best_val   = loaded_val
        best_epoch = ckpt['epoch']

best_val_str = f"{best_val:.4f}" if best_val != float('inf') else "none"
print(f"Current best: epoch {best_epoch}, val_loss={best_val_str}")
print(f"\nLoRA config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}, "
      f"targets={LORA_TARGETS}")
print(f"Augmentation: pitch_shift={'ON' if AUGMENT_PITCH else 'OFF'} (±2 semitones, p=0.3)")
print(f"Training epochs {start_epoch + 1} -> {TARGET_TOTAL_EPOCHS}  "
      f"(warmup={WARMUP_EPOCHS} epoch, early stop patience={EARLY_STOP_PATIENCE})\n")

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'LR':>10} {'Time':>8}")
print("-" * 52)

epochs_no_improve = 0

for epoch in range(start_epoch + 1, TARGET_TOTAL_EPOCHS + 1):
    t0         = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
    val_loss   = eval_epoch(model, val_loader,   DEVICE)
    scheduler.step()
    elapsed    = time.time() - t0
    current_lr = scheduler.get_last_lr()[0]

    lora_config_dict = {
        'r':       LORA_R,
        'alpha':   LORA_ALPHA,
        'dropout': LORA_DROPOUT,
        'targets': list(LORA_TARGETS),
    }

    note = ""
    if val_loss < best_val:
        best_val          = val_loss
        best_epoch        = epoch
        epochs_no_improve = 0
        note              = " ← best"
        torch.save({
            'epoch':           epoch,
            'lora_state_dict': get_lora_state_dict(model),
            'lora_config':     lora_config_dict,
            'train_loss':      train_loss,
            'val_loss':        val_loss,
        }, CHECKPOINT_DIR / 'best_checkpoint.pt')
    else:
        epochs_no_improve += 1

    print(f"{epoch:>6} {train_loss:>10.4f} {val_loss:>10.4f} "
          f"{current_lr:>10.2e} {elapsed:>7.1f}s{note}")

    torch.save({
        'epoch':           epoch,
        'lora_state_dict': get_lora_state_dict(model),
        'lora_config':     lora_config_dict,
        'train_loss':      train_loss,
        'val_loss':        val_loss,
    }, CHECKPOINT_DIR / f'checkpoint_epoch{epoch:03d}.pt')

    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping: val_loss hasn't improved for {EARLY_STOP_PATIENCE} epochs.")
        break

print(f"\nTraining complete! Best: epoch {best_epoch}, val_loss={best_val:.4f}")
print("Run 06_generate_samples_lora.py to create showcase audio.")

# ── Sanity-check generation ──────────────────────────────────────────────────
print("\nGenerating sanity-check samples from best LoRA checkpoint...")
load_lora_checkpoint(model, CHECKPOINT_DIR / 'best_checkpoint.pt', DEVICE)
model.lm.eval()

model.set_generation_params(duration=10, use_sampling=True, top_k=250, temperature=1.0, cfg_coef=3.0)

prompts = [
    "tuvan throat singing, khoomei, drone with overtone melody",
    "mongolian throat singing, kargyraa, deep rumbling voice",
]

with torch.no_grad():
    wavs = model.generate(prompts)

for i, (p, w) in enumerate(zip(prompts, wavs)):
    audio_np = w.squeeze(0).cpu().float().numpy()
    out_path = CHECKPOINT_DIR / f'sanity_{i + 1}.wav'
    sf.write(str(out_path), audio_np, model.sample_rate)
    print(f"  Saved {out_path}  [{p[:50]}]")

print("\nDone. Next: python 06_generate_samples_lora.py")
