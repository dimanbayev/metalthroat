"""
Recover training state from epoch checkpoints.

Reads all checkpoint_epoch*.pt files, extracts loss metadata (no GPU needed),
finds the best epoch, saves best_checkpoint.pt, and regenerates training_curves.png.

Run this after any interrupted training session.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

CHECKPOINT_DIR = Path('checkpoints')
PROGRESS_FILE  = Path('PROGRESS.md')


def load_checkpoint_metadata(path: Path) -> dict:
    """Load only the non-tensor fields from a checkpoint (fast, CPU-only)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return {
        'epoch':      ckpt['epoch'],
        'train_loss': ckpt['train_loss'],
        'val_loss':   ckpt['val_loss'],
        'path':       path,
        'lm_state':   ckpt['lm_state_dict'],  # needed for saving best
    }


def regenerate_curves(records: list, best_epoch: int, out_path: Path):
    epochs      = [r['epoch']      for r in records]
    train_losses = [r['train_loss'] for r in records]
    val_losses   = [r['val_loss']   for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, 'b-o', markersize=4, label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses,   'r-o', markersize=4, label='Val Loss',   linewidth=2)
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                label=f'Best (epoch {best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    gaps = [v - t for t, v in zip(train_losses, val_losses)]
    ax2.plot(epochs, gaps, 'purple', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.fill_between(epochs, gaps, 0,
                     where=[g > 0 for g in gaps], alpha=0.2, color='red',   label='Overfitting')
    ax2.fill_between(epochs, gaps, 0,
                     where=[g <= 0 for g in gaps], alpha=0.2, color='green', label='Underfitting')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val Loss − Train Loss')
    ax2.set_title('Overfitting Indicator')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves -> {out_path}")


def write_progress_md(records: list, best: dict, training_complete: bool = False):
    epochs_done = max(r['epoch'] for r in records)
    target = 30

    lines = [
        "# MetalThroat Training Progress",
        "",
        f"## Status: {'COMPLETE ✅' if training_complete else f'Epoch {epochs_done}/{target} — PAUSED'}",
        f"**Best checkpoint:** `checkpoints/best_checkpoint.pt`  ",
        f"**Best epoch:** {best['epoch']}  ",
        f"**Best val loss:** {best['val_loss']:.4f}  ",
        "",
        "### Loss History",
        "",
        "| Epoch | Train Loss | Val Loss | Best? |",
        "|------:|----------:|--------:|:-----:|",
    ]
    for r in records:
        marker = "✅" if r['epoch'] == best['epoch'] else ""
        lines.append(f"| {r['epoch']:>5} | {r['train_loss']:>10.4f} | {r['val_loss']:>8.4f} | {marker} |")

    lines += [""]

    if not training_complete:
        lines += [
            "### Next Steps",
            "",
            "```bash",
            "# Continue training to epoch 30:",
            "python continue_training.py",
            "```",
            "",
            "Then run evaluation:",
            "```bash",
            "jupyter nbconvert --to notebook --execute 04_evaluation.ipynb",
            "```",
        ]
    else:
        lines += [
            "### Outputs",
            "",
            "- `checkpoints/best_checkpoint.pt` — best fine-tuned LM weights",
            "- `checkpoints/training_curves.png` — full 30-epoch loss curves",
            "- `evaluation/` — base vs fine-tuned comparison WAVs + spectrograms",
            "- `samples/` — showcase throat singing WAVs (best quality)",
        ]

    PROGRESS_FILE.write_text("\n".join(lines), encoding='utf-8')
    print(f"Saved progress -> {PROGRESS_FILE}")


def main():
    print("=" * 55)
    print("  MetalThroat: Checkpoint Recovery")
    print("=" * 55)

    ckpt_files = sorted(CHECKPOINT_DIR.glob('checkpoint_epoch*.pt'))
    if not ckpt_files:
        print("No epoch checkpoints found in checkpoints/")
        return

    print(f"\nFound {len(ckpt_files)} checkpoint(s). Loading metadata...")
    records = []
    for path in ckpt_files:
        try:
            r = load_checkpoint_metadata(path)
            records.append(r)
            print(f"  epoch {r['epoch']:>3}  train={r['train_loss']:.4f}  val={r['val_loss']:.4f}  ({path.name})")
        except Exception as e:
            print(f"  SKIP {path.name}: {e}")

    if not records:
        print("Could not read any checkpoints.")
        return

    records.sort(key=lambda r: r['epoch'])

    all_nan = all(np.isnan(r['val_loss']) for r in records)
    if all_nan:
        print("\nWARNING: All checkpoints have NaN losses.")
        print("  This means training produced NaN gradients (likely autocast dtype mismatch).")
        print("  The weights in these checkpoints are essentially the original pretrained weights.")
        print("  Saving the latest epoch as best_checkpoint.pt as a starting point.")
        print("  The autocast bug has been fixed in continue_training.py.")
        best = records[-1]  # use latest
    else:
        valid = [r for r in records if not np.isnan(r['val_loss'])]
        best  = min(valid, key=lambda r: r['val_loss'])

    val_str = f"{best['val_loss']:.4f}" if not np.isnan(best['val_loss']) else "nan (pretrained weights)"
    print(f"\nBest epoch: {best['epoch']}  val_loss={val_str}")

    # Save best_checkpoint.pt
    best_path = CHECKPOINT_DIR / 'best_checkpoint.pt'
    torch.save({
        'epoch':          best['epoch'],
        'lm_state_dict':  best['lm_state'],
        'train_loss':     best['train_loss'],
        'val_loss':       best['val_loss'],
    }, best_path)
    print(f"Saved best checkpoint -> {best_path}")

    # Regenerate curves (only if we have valid loss data)
    curve_path = CHECKPOINT_DIR / 'training_curves.png'
    if not all_nan:
        regenerate_curves(records, best['epoch'], curve_path)
    else:
        print("Skipping training_curves.png (all losses are NaN — no valid training data yet)")

    # Write PROGRESS.md
    training_complete = (max(r['epoch'] for r in records) >= 30)
    write_progress_md(records, best, training_complete=training_complete)

    print("\nDone. Next step: python continue_training.py")


if __name__ == '__main__':
    main()
