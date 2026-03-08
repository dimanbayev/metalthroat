"""
Continue training from checkpoint - runs for 5 more epochs then generates samples.
"""
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

# Config
DEVICE = 'cuda'
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
ADDITIONAL_EPOCHS = 5
GRAD_CLIP = 1.0
CHECKPOINT_DIR = Path('checkpoints')

# Load dataset config
with open('dataset/dataset_config.json') as f:
    cfg = json.load(f)

DATA_DIR = Path(cfg['data_dir'])
TRAIN_MANIFEST = Path(cfg['train_manifest'])
VAL_MANIFEST = Path(cfg['val_manifest'])
SAMPLE_RATE = cfg['sample_rate']
CLIP_DURATION = cfg['clip_duration']

class ThroatSingingDataset(Dataset):
    def __init__(self, manifest_path, data_dir, sample_rate=32000, clip_duration=10.0):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * clip_duration)
        self.records = []
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        print(f"Loaded {len(self.records)} records from {Path(manifest_path).name}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        audio_np, sr = sf.read(str(self.data_dir / record['path']))
        audio = torch.from_numpy(audio_np).float()
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
    logits_flat = logits.reshape(-1, card)
    codes_flat = codes.reshape(-1)
    mask_flat = mask.reshape(-1)
    loss_per_token = F.cross_entropy(logits_flat, codes_flat, reduction='none')
    masked_loss = loss_per_token * mask_flat.float()
    return masked_loss.sum() / (mask_flat.sum() + 1e-8)

def train_epoch(model, loader, optimizer, device):
    model.lm.train()
    total_loss = 0
    for audio, descs in loader:
        audio = audio.to(device)
        with torch.no_grad():
            codes, _ = model.compression_model.encode(audio)
        conditions = [ConditioningAttributes(text={'description': d}) for d in descs]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model.lm.compute_predictions(codes=codes, conditions=conditions)
            loss = compute_loss(out.logits, codes, out.mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lm.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.lm.eval()
    total_loss = 0
    for audio, descs in loader:
        audio = audio.to(device)
        codes, _ = model.compression_model.encode(audio)
        conditions = [ConditioningAttributes(text={'description': d}) for d in descs]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model.lm.compute_predictions(codes=codes, conditions=conditions)
            loss = compute_loss(out.logits, codes, out.mask)
        total_loss += loss.item()
    return total_loss / len(loader)

print("Loading datasets...")
train_ds = ThroatSingingDataset(TRAIN_MANIFEST, DATA_DIR, SAMPLE_RATE, CLIP_DURATION)
val_ds = ThroatSingingDataset(VAL_MANIFEST, DATA_DIR, SAMPLE_RATE, CLIP_DURATION)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Loading model...")
model = MusicGen.get_pretrained('facebook/musicgen-small', device=DEVICE)

# Freeze EnCodec
for p in model.compression_model.parameters():
    p.requires_grad = False
model.compression_model.eval()

# Load checkpoint
ckpt_path = CHECKPOINT_DIR / 'checkpoint_epoch005.pt'
if ckpt_path.exists():
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.lm.load_state_dict(ckpt['lm_state_dict'])
    start_epoch = ckpt['epoch']
    print(f"Resuming from epoch {start_epoch}, train_loss={ckpt['train_loss']:.4f}, val_loss={ckpt['val_loss']:.4f}")
else:
    start_epoch = 0
    print("No checkpoint found, starting fresh")

# Setup optimizer
optimizer = AdamW(model.lm.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.01)
total_epochs = start_epoch + ADDITIONAL_EPOCHS
scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

# Skip scheduler steps for completed epochs
for _ in range(start_epoch):
    scheduler.step()

best_val = float('inf')
best_epoch = 0

print(f"\nTraining epochs {start_epoch+1} to {total_epochs}")
print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'Time':>8}")
print("-" * 40)

for epoch in range(start_epoch + 1, total_epochs + 1):
    t0 = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
    val_loss = eval_epoch(model, val_loader, DEVICE)
    scheduler.step()
    elapsed = time.time() - t0

    note = ""
    if val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch
        note = " *best*"
        torch.save({
            'epoch': epoch,
            'lm_state_dict': model.lm.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, CHECKPOINT_DIR / 'best_checkpoint.pt')

    print(f"{epoch:>6} {train_loss:>10.4f} {val_loss:>10.4f} {elapsed:>7.1f}s{note}")

    # Save periodic checkpoint
    torch.save({
        'epoch': epoch,
        'lm_state_dict': model.lm.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, CHECKPOINT_DIR / f'checkpoint_epoch{epoch:03d}.pt')

print(f"\nTraining complete! Best: epoch {best_epoch}, val_loss={best_val:.4f}")

# Generate samples
print("\nGenerating test samples...")
ckpt = torch.load(CHECKPOINT_DIR / 'best_checkpoint.pt', map_location=DEVICE)
model.lm.load_state_dict(ckpt['lm_state_dict'])
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
    sf.write(f'checkpoints/finetuned_sample_{i+1}.wav', audio_np, model.sample_rate)
    print(f"Saved: checkpoints/finetuned_sample_{i+1}.wav")
    print(f"  Prompt: {p}")

print("\nDone!")
