"""
Continue training from the latest epoch checkpoint.

Auto-detects the most recent checkpoint_epoch*.pt, resumes training,
and runs until epoch TARGET_TOTAL_EPOCHS (default: 30).
Saves best_checkpoint.pt whenever val_loss improves.

Usage:
    python continue_training.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ConditioningAttributes
import time
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
TARGET_TOTAL_EPOCHS = 30     # Run until this epoch (script is a no-op if already there)
BATCH_SIZE          = 4
LEARNING_RATE       = 1e-5
GRAD_CLIP           = 1.0
DEVICE              = 'cuda'
CHECKPOINT_DIR      = Path('checkpoints')

# ── Dataset ──────────────────────────────────────────────────────────────────
with open('dataset/dataset_config.json') as f:
    cfg = json.load(f)

DATA_DIR       = Path(cfg['data_dir'])
TRAIN_MANIFEST = Path(cfg['train_manifest'])
VAL_MANIFEST   = Path(cfg['val_manifest'])
SAMPLE_RATE    = cfg['sample_rate']
CLIP_DURATION  = cfg['clip_duration']


class ThroatSingingDataset(Dataset):
    def __init__(self, manifest_path, data_dir, sample_rate=32000, clip_duration=10.0):
        self.data_dir    = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * clip_duration)
        self.records     = []
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        print(f"Loaded {len(self.records)} records from {Path(manifest_path).name}")

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
        return audio, record['description']


def compute_loss(logits, codes, mask):
    B, K, T, card = logits.shape
    # MusicGen uses a codebook delay pattern: codebook k has k undefined (NaN) positions
    # at the start. The mask correctly excludes them, but NaN * 0 = NaN in IEEE 754.
    # Replace NaN with 0 before cross_entropy so masking works correctly.
    logits         = logits.nan_to_num(nan=0.0)
    logits_flat    = logits.reshape(-1, card)
    codes_flat     = codes.reshape(-1)
    mask_flat      = mask.reshape(-1)
    loss_per_token = F.cross_entropy(logits_flat, codes_flat, reduction='none')
    masked_loss    = loss_per_token * mask_flat.float()
    return masked_loss.sum() / (mask_flat.sum() + 1e-8)


def train_epoch(model, loader, optimizer, device):
    model.lm.train()
    total_loss  = 0
    num_batches = 0
    nan_batches = 0
    for audio, descs in tqdm(loader, desc="train", leave=False, dynamic_ncols=True):
        audio      = audio.to(device)
        with torch.no_grad():
            codes, _ = model.compression_model.encode(audio)
        conditions = [ConditioningAttributes(text={'description': d}) for d in descs]
        # Autocast is deliberately DISABLED for stability.
        # audiocraft 1.3.0 has dtype inconsistencies between LM layers; autocast
        # reintroduces float16/bfloat16 mismatches that cause NaN loss.
        # We use float32 throughout (model.lm was cast above).
        out  = model.lm.compute_predictions(codes=codes, conditions=conditions)
        loss = compute_loss(out.logits, codes, out.mask)
        if torch.isnan(loss):
            nan_batches += 1
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lm.parameters(), GRAD_CLIP)
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
    for audio, descs in tqdm(loader, desc="val  ", leave=False, dynamic_ncols=True):
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


# ── Auto-detect latest checkpoint ────────────────────────────────────────────
all_ckpts = sorted(CHECKPOINT_DIR.glob('checkpoint_epoch*.pt'))
if not all_ckpts:
    print("No epoch checkpoints found — starting from scratch")
    ckpt_path = None
else:
    ckpt_path = all_ckpts[-1]   # highest epoch number (alphabetical sort = epoch order)
    print(f"Latest checkpoint: {ckpt_path.name}")

# ── Datasets & loaders ───────────────────────────────────────────────────────
print("\nLoading datasets...")
train_ds = ThroatSingingDataset(TRAIN_MANIFEST, DATA_DIR, SAMPLE_RATE, CLIP_DURATION)
val_ds   = ThroatSingingDataset(VAL_MANIFEST,   DATA_DIR, SAMPLE_RATE, CLIP_DURATION)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading MusicGen-small...")
model = MusicGen.get_pretrained('facebook/musicgen-small', device=DEVICE)
model.lm = model.lm.float()

for p in model.compression_model.parameters():
    p.requires_grad = False
model.compression_model.eval()

# ── Resume from checkpoint ───────────────────────────────────────────────────
if ckpt_path and ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.lm.load_state_dict(ckpt['lm_state_dict'])
    start_epoch = ckpt['epoch']
    print(f"Resumed from epoch {start_epoch}  "
          f"train={ckpt['train_loss']:.4f}  val={ckpt['val_loss']:.4f}")
else:
    start_epoch = 0
    print("Starting fresh from pretrained weights")

if start_epoch >= TARGET_TOTAL_EPOCHS:
    print(f"\nAlready at epoch {start_epoch} >= {TARGET_TOTAL_EPOCHS}. Nothing to do.")
    print("Run generate_samples.py to create showcase audio.")
    exit(0)

# ── Optimizer & scheduler ────────────────────────────────────────────────────
optimizer = AdamW(model.lm.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=TARGET_TOTAL_EPOCHS, eta_min=1e-6)
# Fast-forward scheduler to match where we are in the cosine curve
for _ in range(start_epoch):
    scheduler.step()

best_val   = float('inf')
best_epoch = start_epoch

# Seed best_val from already-loaded checkpoint (avoids reloading all ~7GB of checkpoints)
if ckpt_path and ckpt_path.exists():
    loaded_val = ckpt['val_loss']
    if not (loaded_val != loaded_val):  # not NaN check
        best_val   = loaded_val
        best_epoch = ckpt['epoch']
best_val_str = f"{best_val:.4f}" if best_val != float('inf') else "none (all NaN)"
print(f"Current best: epoch {best_epoch}, val_loss={best_val_str}\n")

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"Training epochs {start_epoch + 1} -> {TARGET_TOTAL_EPOCHS}")
print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'LR':>10} {'Time':>8}")
print("-" * 52)

for epoch in range(start_epoch + 1, TARGET_TOTAL_EPOCHS + 1):
    t0         = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
    val_loss   = eval_epoch(model, val_loader,   DEVICE)
    scheduler.step()
    elapsed    = time.time() - t0
    current_lr = scheduler.get_last_lr()[0]

    note = ""
    if val_loss < best_val:
        best_val   = val_loss
        best_epoch = epoch
        note       = " ← best"
        torch.save({
            'epoch':         epoch,
            'lm_state_dict': model.lm.state_dict(),
            'train_loss':    train_loss,
            'val_loss':      val_loss,
        }, CHECKPOINT_DIR / 'best_checkpoint.pt')

    print(f"{epoch:>6} {train_loss:>10.4f} {val_loss:>10.4f} {current_lr:>10.2e} {elapsed:>7.1f}s{note}")

    torch.save({
        'epoch':         epoch,
        'lm_state_dict': model.lm.state_dict(),
        'train_loss':    train_loss,
        'val_loss':      val_loss,
    }, CHECKPOINT_DIR / f'checkpoint_epoch{epoch:03d}.pt')

print(f"\nTraining complete! Best: epoch {best_epoch}, val_loss={best_val:.4f}")
print("Run recover_state.py to regenerate training_curves.png and PROGRESS.md")

# ── Sanity-check generation ──────────────────────────────────────────────────
print("\nGenerating sanity-check samples from best checkpoint...")
ckpt = torch.load(CHECKPOINT_DIR / 'best_checkpoint.pt', map_location=DEVICE, weights_only=False)
model.lm.load_state_dict(ckpt['lm_state_dict'])
model.lm.eval()

model.set_generation_params(duration=10, use_sampling=True, top_k=250, temperature=1.0, cfg_coef=3.0)

prompts = [
    "tuvan throat singing, khoomei, drone with overtone melody",
    "mongolian throat singing, kargyraa, deep rumbling voice",
]

with torch.no_grad():
    wavs = model.generate(prompts)

CHECKPOINT_DIR.mkdir(exist_ok=True)
for i, (p, w) in enumerate(zip(prompts, wavs)):
    audio_np = w.squeeze(0).cpu().float().numpy()
    out_path = CHECKPOINT_DIR / f'sanity_{i + 1}.wav'
    sf.write(str(out_path), audio_np, model.sample_rate)
    print(f"  Saved {out_path}  [{p[:50]}]")

print("\nDone. Next: python generate_samples.py")
