"""
Diagnose the NaN loss issue during training.
Loads one batch and checks for NaN at every step.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import soundfile as sf
import torch.nn.functional as F
from pathlib import Path
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ConditioningAttributes

DEVICE = 'cuda'

print("Loading MusicGen-small...")
model = MusicGen.get_pretrained('facebook/musicgen-small', device=DEVICE)

print("\n--- Before float() conversion ---")
for name, p in list(model.lm.named_parameters())[:3]:
    print(f"  {name}: {p.dtype}")

print("\nConverting LM to float32...")
model.lm = model.lm.float()

print("\n--- After float() conversion ---")
for name, p in list(model.lm.named_parameters())[:3]:
    print(f"  {name}: {p.dtype}")

# Also check if there are any buffers that stayed in other dtypes
print("\n--- LM buffers (non-parameter tensors) ---")
for name, buf in list(model.lm.named_buffers())[:5]:
    print(f"  {name}: {buf.dtype}, shape={list(buf.shape)[:3]}")

# Freeze EnCodec
for p in model.compression_model.parameters():
    p.requires_grad = False
model.compression_model.eval()
model.lm.train()

# Load one real audio sample
print("\nLoading one training sample...")
with open('dataset/train.jsonl') as f:
    record = json.loads(f.readline())

data_dir = Path('dataset/processed')
audio_np, sr = sf.read(str(data_dir / record['path']))
audio = torch.from_numpy(audio_np).float()
if audio.dim() == 1:
    audio = audio.unsqueeze(0)
audio = audio[:, :320000]  # 10s at 32kHz
audio = audio.unsqueeze(0).to(DEVICE)  # [1, 1, T]

print(f"  Audio shape: {audio.shape}, dtype: {audio.dtype}")
print(f"  Audio NaN: {torch.isnan(audio).any().item()}, Inf: {torch.isinf(audio).any().item()}")
print(f"  Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

# Step 1: Encode with EnCodec
print("\nStep 1: EnCodec encode...")
with torch.no_grad():
    codes, scale = model.compression_model.encode(audio)
print(f"  Codes shape: {codes.shape}, dtype: {codes.dtype}")
print(f"  Codes range: [{codes.min().item()}, {codes.max().item()}]")
print(f"  Codes NaN: {torch.isnan(codes.float()).any().item()}")

# Step 2: Conditions
print("\nStep 2: Build conditions...")
conditions = [ConditioningAttributes(text={'description': record['description']})]
print(f"  Description: '{record['description'][:60]}'")

# Step 3: compute_predictions
print("\nStep 3: LM compute_predictions (training mode)...")
try:
    out = model.lm.compute_predictions(codes=codes, conditions=conditions)
    print(f"  Logits shape: {out.logits.shape}, dtype: {out.logits.dtype}")
    print(f"  Logits NaN: {torch.isnan(out.logits).any().item()}")
    print(f"  Logits Inf: {torch.isinf(out.logits).any().item()}")
    if torch.isnan(out.logits).any():
        total = out.logits.numel()
        nan_count = torch.isnan(out.logits).sum().item()
        print(f"  NaN fraction: {nan_count}/{total} = {nan_count/total:.1%}")
        # Check which codebooks have NaN
        for k in range(out.logits.shape[1]):
            k_nan = torch.isnan(out.logits[:, k]).sum().item()
            print(f"    Codebook {k}: {k_nan} NaN values")
    else:
        print(f"  Logits range: [{out.logits.min():.3f}, {out.logits.max():.3f}]")
except Exception as e:
    print(f"  ERROR in compute_predictions: {e}")
    raise

# Step 4: Loss
print("\nStep 4: Computing loss...")
B, K, T, card = out.logits.shape
logits_flat = out.logits.reshape(-1, card)
codes_flat = codes.reshape(-1)
mask_flat = out.mask.reshape(-1)
loss_raw = F.cross_entropy(logits_flat, codes_flat, reduction='none')
print(f"  Per-token loss NaN: {torch.isnan(loss_raw).any().item()}")
masked_loss = (loss_raw * mask_flat.float()).sum() / (mask_flat.sum() + 1e-8)
print(f"  Final loss: {masked_loss.item()}")

# Check if conditioning caused the NaN
print("\nStep 5: Test with inference mode (for comparison)...")
model.lm.eval()
with torch.no_grad():
    out_eval = model.lm.compute_predictions(codes=codes, conditions=conditions)
print(f"  [EVAL] Logits NaN: {torch.isnan(out_eval.logits).any().item()}")
if not torch.isnan(out_eval.logits).any():
    print(f"  [EVAL] Logits range: [{out_eval.logits.min():.3f}, {out_eval.logits.max():.3f}]")
    print("  --> NaN only appears in TRAIN mode, not EVAL mode")
    print("      This suggests dropout or training-specific layers are the cause")
else:
    print("  --> NaN also in EVAL mode — not a dropout issue")

print("\nDiagnosis complete.")
