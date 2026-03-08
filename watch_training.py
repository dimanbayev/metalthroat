"""
Live training monitor — polls checkpoints every 30s and prints a status table.
Works with a training process that is already running; no restart needed.

Usage (in a separate terminal):
    python watch_training.py
    python watch_training.py --interval 60   # poll every 60s
    python watch_training.py --once          # print once and exit
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

CHECKPOINT_DIR = Path('checkpoints')
TARGET_EPOCHS  = 30


def load_meta(ckpt_path: Path) -> dict | None:
    """Load only scalar metadata from a checkpoint (CPU, no model weights needed)."""
    try:
        data = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        return {
            'epoch':      data['epoch'],
            'train_loss': data['train_loss'],
            'val_loss':   data['val_loss'],
            'mtime':      ckpt_path.stat().st_mtime,
        }
    except Exception:
        return None


def render(records: list[dict], best: dict | None, secs_per_epoch: float | None):
    now = time.strftime('%H:%M:%S')
    current_epoch = records[-1]['epoch'] if records else 0

    # ETA
    if secs_per_epoch and current_epoch < TARGET_EPOCHS:
        remaining = (TARGET_EPOCHS - current_epoch) * secs_per_epoch
        h, rem = divmod(int(remaining), 3600)
        m, s   = divmod(rem, 60)
        eta_str = f"{h:02d}:{m:02d}:{s:02d}"
        rate_str = f"{secs_per_epoch:.0f}s/epoch"
    else:
        eta_str  = "—"
        rate_str = "—"

    # Header
    lines = [
        f"MetalThroat  |  {now}  |  epoch {current_epoch}/{TARGET_EPOCHS}",
        f"rate: {rate_str}   ETA: {eta_str}",
        "",
        f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}",
        "─" * 32,
    ]

    # Show last 10 epochs
    for r in records[-10:]:
        marker = " ←best" if best and r['epoch'] == best['epoch'] else ""
        lines.append(
            f"{r['epoch']:>6}  {r['train_loss']:>10.4f}  {r['val_loss']:>10.4f}{marker}"
        )

    if best:
        lines += [
            "─" * 32,
            f"Best: epoch {best['epoch']}  val={best['val_loss']:.4f}",
        ]

    lines.append("\n(Ctrl+C to quit)")
    return "\n".join(lines)


def refresh():
    ckpts = sorted(CHECKPOINT_DIR.glob('checkpoint_epoch*.pt'))
    if not ckpts:
        print("No checkpoints found yet. Waiting...")
        return

    records = []
    for p in ckpts:
        m = load_meta(p)
        if m:
            records.append(m)

    if not records:
        print("Checkpoints exist but couldn't read metadata.")
        return

    # Best by val loss
    best = min(records, key=lambda r: r['val_loss'])

    # Epoch rate from last two checkpoint timestamps
    secs_per_epoch = None
    if len(ckpts) >= 2:
        t1 = ckpts[-2].stat().st_mtime
        t2 = ckpts[-1].stat().st_mtime
        diff = t2 - t1
        if diff > 0:
            secs_per_epoch = diff

    os.system('cls' if sys.platform == 'win32' else 'clear')
    print(render(records, best, secs_per_epoch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=30, help='Poll interval in seconds')
    parser.add_argument('--once', action='store_true', help='Print once and exit')
    args = parser.parse_args()

    print(f"Watching {CHECKPOINT_DIR.resolve()} ...")
    if args.once:
        refresh()
        return

    try:
        while True:
            refresh()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == '__main__':
    main()
