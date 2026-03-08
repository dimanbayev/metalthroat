# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MetalThroat is a Jupyter notebook-based ML project for fine-tuning Meta's MusicGen audio generation model to produce Mongolian/Tuvan throat singing (khoomei, sygyt, kargyraa). It demonstrates domain adaptation of a general music generation model to specialize in an underrepresented audio style.

## Environment Setup

```bash
# Prerequisites: Python 3.10+, ffmpeg installed

# RTX 5090/Blackwell (CUDA 12.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Older GPUs (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Running the Project

Execute notebooks sequentially:
1. `01_setup_and_inference.ipynb` - Environment validation, load MusicGen-small, generate baseline samples
2. `02_data_preparation.ipynb` - YouTube data collection, audio preprocessing, dataset manifest creation
3. `03_finetuning.ipynb` - Training loop with frozen EnCodec, trainable LM
4. `04_evaluation.ipynb` - Side-by-side comparison of base vs fine-tuned outputs

```bash
jupyter notebook
```

## Architecture

**MusicGen Model Stack:**
```
Text Input → [T5 Text Encoder (FROZEN)] → [Transformer LM (FINE-TUNED)] → [EnCodec (FROZEN)] → Audio
```

- Audio is tokenized by EnCodec (~50 tokens/second, 4 codebooks × 2048 vocab)
- Only the Transformer Language Model is trained; EnCodec and T5 stay frozen
- 10-second clips = ~2,000 tokens; cross-entropy loss over token prediction

## Key Paths

- `dataset/raw/` - Downloaded YouTube audio
- `dataset/processed/` - Segmented 10s WAV clips at 32kHz
- `dataset/train.jsonl`, `dataset/val.jsonl` - Training manifests
- `checkpoints/best_checkpoint.pt` - Trained model weights

## Training Configuration (Notebook 3)

- Batch size: 4
- Learning rate: 1e-5 (conservative to prevent catastrophic forgetting)
- Optimizer: AdamW (betas=0.9/0.95, weight_decay=0.01)
- Scheduler: CosineAnnealingLR over 30 epochs
- Early stopping patience: 10 epochs
- Mixed precision: bfloat16

## Troubleshooting

- **GPU OOM**: Reduce batch size in NB3
- **Poor results**: Check data quality in NB2, increase dataset size
- **Catastrophic forgetting**: Lower learning rate
- **YouTube download failures**: Update yt-dlp, verify ffmpeg installation
