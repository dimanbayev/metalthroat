# MetalThroat v2: LoRA Experiment

**Branch:** `lora-experiment`
**Status:** Ready to train
**Base model:** `facebook/musicgen-small` (300M params)
**Goal:** Fix the garbled, low-quality output from v1 full fine-tuning

---

## What This Experiment Is

The v1 full fine-tune (30 epochs, all 300M LM weights updated) produced garbled audio. The cause is clear from the loss curves: catastrophic overfitting and partial forgetting of MusicGen's audio generation priors.

| Epoch | Train Loss | Val Loss |
|------:|----------:|--------:|
| 12 (best) | 1.5825 | 2.5644 |
| 30 (final) | 0.4986 | 3.2940 |

The train/val gap at the best epoch is ~1.0 nats. For a 11-hour dataset, this means the model memorized the specific token sequences in the training clips rather than learning to generate throat singing in general.

**LoRA fixes this** by constraining all weight updates to a low-rank subspace (~44M parameters out of 420M). The 376M frozen base weights preserve MusicGen's harmonic modeling, timbral coherence, and rhythmic grammar. Only the domain adaptation direction is learned.

---

## Architecture

### What Changes vs Base MusicGen

```
Base MusicGen:
  T5 (frozen) → Transformer LM (frozen) → EnCodec (frozen)
                    ↑ 300M params, all frozen

v2 LoRA:
  T5 (frozen) → Transformer LM (frozen + LoRA adapters) → EnCodec (frozen)
                    ↑ 376M frozen, 44M LoRA trainable
```

### Which Layers Get LoRA Adapters

Each of the 24 transformer layers has 3 `nn.Linear` modules targeted:

| Module | Shape | Role |
|---|---|---|
| `out_proj` (self_attn) | [1024, 1024] | Attention output projection |
| `out_proj` (cross_attn) | [1024, 1024] | Cross-attention output (text conditioning) |
| `linear1` | [4096, 1024] | FFN up-projection |
| `linear2` | [1024, 4096] | FFN down-projection |

**Why these three (not `in_proj`)?**
MusicGen's QKV projection is stored as a single fused `nn.Parameter` called `in_proj_weight` — not an `nn.Linear` submodule. It cannot be detected or replaced by module-scanning code. `out_proj`, `linear1`, and `linear2` are proper `nn.Linear` submodules and are the standard LoRA targets for transformer models.

### Why Manual LoRA (not HuggingFace PEFT)

PEFT's `get_peft_model()` wraps `model.lm` in a `PeftModel` class. `PeftModel` does not have a `compute_predictions()` method. The entire training/inference pipeline depends on `model.lm.compute_predictions(codes, conditions)` — this would fail immediately with `AttributeError`.

Manual LoRA (~80 lines in `lora_utils.py`) keeps the `LMModel` interface 100% intact and adds zero new dependencies.

### Parameter Count

```
out_proj (self + cross) × 24 layers: 2 × 2 × (1024 + 1024) × 128 = 12,582,912
linear1 × 24 layers:                 2 × (1024 + 4096) × 128     = 15,728,640
linear2 × 24 layers:                 2 × (4096 + 1024) × 128     = 15,728,640
─────────────────────────────────────────────────────────────────────────────
Total trainable:                                                   ~44,040,192
Total LM params:                                                   ~420,000,000
Trainable fraction:                                                ~10.5%
```

This ~40M target is directly motivated by arXiv:2506.21298 which studied Hindustani and Turkish Makam (same low-resource non-Western tradition problem) and found 40M parameters as the optimal adapter size.

---

## Key ML Improvements Over Vanilla LoRA

### 1. LoRA Dropout (p=0.05)
Applied between `lora_A` and `lora_B` in `LoRALinear.forward()`. Motivated by arXiv:2404.09610 which proves LoRA Dropout outperforms vanilla LoRA on small datasets via sparsity regularization. With 11 hours of audio, this is a clear win.

### 2. LR Warmup (1 epoch)
`SequentialLR` with `LinearLR` (LR/1000 → LR over 1 epoch) then `CosineAnnealingLR`. MusicGen's own training uses cosine warmup. Starting cold at 3e-4 on audio token prediction — where the loss landscape is sharper than text LMs — risks early gradient instability.

### 3. Pitch-Shift Augmentation (±2 semitones, 30% of training clips)
Applied in `ThroatSingingDataset.__getitem__` using `torchaudio.functional.pitch_shift` (already installed, zero new deps). Diversifies the harmonic distribution without distorting the drone-overtone relationships characteristic of throat singing. **Never applied to validation.**

---

## File Map

| File | Purpose |
|---|---|
| `lora_utils.py` | Core LoRA primitives: `LoRALinear`, `inject_lora()`, `freeze_base_model()`, `load_lora_checkpoint()`, `get_lora_state_dict()`, `count_trainable()` |
| `05_train_lora.py` | Full training script — runs 60 epochs with early stopping, saves adapter-only checkpoints (~176 MB each vs ~1.6 GB for v1) |
| `06_generate_samples_lora.py` | Inference script — parameter grid search, saves best samples to `samples_lora/` |
| `05_lora_evaluation.ipynb` | 3-way comparison: base MusicGen vs v1 full fine-tune vs v2 LoRA (audio widgets + spectrograms + metrics) |
| `checkpoints_lora/` | LoRA adapter checkpoints (epoch + best). Do not mix with `checkpoints/` (v1 full LM checkpoints) |
| `samples_lora/` | Generated showcase WAVs from v2 LoRA model |

---

## Workflow

### Step 1: Smoke Test (2 epochs, ~3 minutes)

Open `05_train_lora.py`, temporarily set `TARGET_TOTAL_EPOCHS = 2`, then run:

```bash
python 05_train_lora.py
```

Verify:
- `Trainable LM params: 44,040,192 / ...` is printed at startup
- Epoch 1 LR is very low (warmup): should be ~3e-7 at start, rising to ~3e-4
- Epoch 1 and 2 losses are real numbers, not NaN
- `checkpoints_lora/checkpoint_epoch001.pt` exists and is ~176 MB
- Checkpoint contains `lora_config: {r: 128, alpha: 256, dropout: 0.05, targets: [...]}`

Restore `TARGET_TOTAL_EPOCHS = 60`.

### Step 2: Full Training Run (~2–3 hours)

```bash
python 05_train_lora.py
```

Expected behavior:
- Val loss should track train loss more closely than v1 (smaller gap = less overfitting)
- Best epoch likely in 15–30 range (vs epoch 12 in v1)
- Early stopping if no improvement for 15 epochs

### Step 3: Generate Showcase Samples

```bash
python 06_generate_samples_lora.py
```

Outputs: `samples_lora/best_khoomei.wav`, `best_kargyraa.wav`, `best_sygyt.wav`, `showcase_30s.wav`

### Step 4: Evaluate (3-way comparison)

```bash
jupyter notebook 05_lora_evaluation.ipynb
```

Generates spectrograms, harmonic ratio bar charts, and audio widgets comparing:
- Base MusicGen (no fine-tuning)
- v1 Full fine-tune (best epoch 12 from `checkpoints/best_checkpoint.pt`)
- v2 LoRA (best epoch from `checkpoints_lora/best_checkpoint.pt`)

---

## Hyperparameter Rationale

| Parameter | Value | Rationale |
|---|---|---|
| `LORA_R` | 128 | Chosen to yield ~44M trainable params — the optimal adapter size per arXiv:2506.21298 |
| `LORA_ALPHA` | 256 | Scaling = alpha/r = 2.0 — standard LoRA convention (alpha = 2× rank) |
| `LORA_DROPOUT` | 0.05 | Conservative sparsity regularization per arXiv:2404.09610; enough to regularize, not enough to destabilize |
| `LEARNING_RATE` | 3e-4 | Standard LoRA LR (~30× higher than full fine-tune); adapters train in a constrained subspace so higher LR is appropriate |
| `WARMUP_EPOCHS` | 1 | ~886 steps of linear warmup (LR/1000 → LR); prevents early instability |
| `TARGET_TOTAL_EPOCHS` | 60 | 2× v1; LoRA generalizes better so it can run longer before overfitting |
| `EARLY_STOP_PATIENCE` | 15 | More generous than v1 (10); gives LoRA room to find a better basin |
| `weight_decay` | 0.0 | Not applied to LoRA adapters — the low-rank constraint is already a form of regularization |
| `AUGMENT_PITCH` | True | ±2 semitones, 30% of clips; diversifies harmonic distribution without breaking timbre |
| `BATCH_SIZE` | 4 | Same as v1; LoRA doesn't significantly change memory footprint |

---

## Known Issues / Audiocraft Gotchas

These apply to ALL training in this project (v1 and v2):

1. **No autocast.** `torch.autocast` with `bfloat16` on audiocraft 1.3.0 causes NaN losses. The LM must be kept in `float32` at all times (`model.lm = model.lm.float()`). No `with torch.autocast():` blocks anywhere.

2. **`nan_to_num` before cross_entropy.** MusicGen's codebook delay pattern produces NaN logits for the first K positions of codebook K. The `lm_output.mask` correctly excludes them, but `NaN × 0 = NaN` in IEEE 754. Always call `logits.nan_to_num(nan=0.0)` before computing cross-entropy.

3. **`in_proj_weight` is not an `nn.Linear`.** MusicGen fuses QKV into a single `nn.Parameter`. Any code that tries to target `q_proj`, `k_proj`, or `v_proj` will silently fail (the module scanner won't find them). We target `out_proj`, `linear1`, `linear2` only.

4. **Checkpoint format mismatch.** v1 checkpoints store `lm_state_dict` (full LM weights ~1.6 GB). v2 checkpoints store `lora_state_dict` (LoRA weights only, ~176 MB). These are incompatible — do not pass a v1 checkpoint to `load_lora_checkpoint()` or vice versa.

5. **`compression_model` must stay in eval mode.** Call `model.compression_model.eval()` after every model load. If it accidentally enters train mode it will update BatchNorm statistics and produce inconsistent tokenization.

---

## Future Experiments

### rsLoRA (Rank-Stabilized LoRA) — arXiv:2312.03732
Change `LoRALinear` scaling from `alpha/r` to `alpha/sqrt(r)`. At r=128 this increases the scaling from 2.0 to ~22.6 — an 11× jump that requires reducing `LEARNING_RATE` to ~1e-4. The `use_rslora` flag is already wired into `lora_utils.py`; just set it to `True` and reduce LR accordingly. Best for comparing at multiple ranks to find the optimal r.

### DoRA (Weight-Decomposed LoRA) — arXiv:2402.09353
Decomposes the weight update into magnitude and direction components, applying LoRA only to the direction. Consistently outperforms vanilla LoRA on language and vision tasks, has zero inference overhead (merges back after training). Implementation is moderate complexity (~50 additional lines). Worth trying after a baseline LoRA result exists.

### CLAP Score Evaluation
CLAP (Contrastive Language-Audio Pretraining) measures text-audio semantic alignment as cosine similarity between audio and text embeddings. More objective than harmonic ratio for measuring whether the model actually responded to the text prompt. Requires loading a separate CLAP model (e.g., from `laion-ai/clap`). Add to `05_lora_evaluation.ipynb` after baseline results.

### MusicGen-Medium (1.5B params)
Swap `facebook/musicgen-small` for `facebook/musicgen-medium`. With LoRA, VRAM is well within 32 GB (RTX 5090). The medium model has 5× the parameters and substantially better audio priors. The argument for doing LoRA-small first: confirm LoRA itself helps before adding the confound of model size.

### More Training Data
The dataset has ~11 hours. Doubling to 20+ hours by downloading more YouTube content (yt-dlp is already in the pipeline from NB2) would be the highest-leverage change if LoRA results still plateau early. Aim for more diversity within each style (more singers, more recording conditions).

### MusicGen-Melody with Audio Conditioning
The `facebook/musicgen-melody` variant accepts an audio reference alongside the text prompt. This opens the possibility of conditioning generation on a reference throat singing clip, which would strongly constrain the output toward the correct timbre. Requires a different data format (melody conditioning instead of text-only).
