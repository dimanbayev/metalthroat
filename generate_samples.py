"""
Generate best-quality throat singing showcase samples.

For each prompt, generates N candidates across a parameter grid and
keeps the one with the highest harmonic ratio (most tonal = most throat-singing-like).

Output: samples/best_khoomei.wav, best_kargyraa.wav, best_sygyt.wav
        samples/showcase_30s.wav  (30-second version of the best-scoring style)

Usage:
    python generate_samples.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import soundfile as sf
import torch
import librosa
from pathlib import Path
from audiocraft.models import MusicGen

DEVICE        = 'cuda'
CHECKPOINT    = Path('checkpoints/best_checkpoint.pt')
SAMPLES_DIR   = Path('samples')
SAMPLES_DIR.mkdir(exist_ok=True)

# ── Showcase prompts — matched to training vocabulary ────────────────────────
SHOWCASE_PROMPTS = [
    {
        "label":  "khoomei",
        "prompt": "tuvan throat singing, khoomei style, meditative drone with overtone harmonics, traditional, ancient",
    },
    {
        "label":  "kargyraa",
        "prompt": "mongolian throat singing, kargyraa, deep subharmonic drone, ceremonial, low voice, ancient",
    },
    {
        "label":  "sygyt",
        "prompt": "sygyt throat singing, high overtone flute melody over sustained drone, tuvan folk music",
    },
]

# ── Inference parameter grid (4 candidates per prompt) ───────────────────────
PARAM_GRID = [
    dict(temperature=0.90, cfg_coef=4.0, top_k=250),   # tight + strong guidance
    dict(temperature=1.00, cfg_coef=3.5, top_k=250),   # balanced
    dict(temperature=0.85, cfg_coef=5.0, top_k=200),   # very tight, max guidance
    dict(temperature=1.00, cfg_coef=3.0, top_k=500),   # more diverse
]

DURATION_SHORT   = 15   # seconds — for each style's best sample
DURATION_SHOWCASE = 30  # seconds — one longer showpiece
SAMPLE_RATE      = 32000


def harmonic_ratio(audio_np: np.ndarray, sr: int) -> float:
    """Fraction of energy that is harmonic (higher = more tonal, better for throat singing)."""
    harmonic, _ = librosa.effects.hpss(audio_np)
    return float(np.mean(harmonic ** 2) / (np.mean(audio_np ** 2) + 1e-10))


def generate_candidates(model, prompt: str, params: dict, duration: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.set_generation_params(
        duration=duration,
        use_sampling=True,
        top_k=params['top_k'],
        temperature=params['temperature'],
        cfg_coef=params['cfg_coef'],
    )
    with torch.no_grad():
        wav = model.generate([prompt])  # [1, 1, T]
    return wav[0].squeeze(0).cpu().float().numpy()


def main():
    print("=" * 60)
    print("  MetalThroat: Best-Quality Sample Generation")
    print("=" * 60)

    if not CHECKPOINT.exists():
        print(f"ERROR: {CHECKPOINT} not found. Run recover_state.py first.")
        return

    print(f"\nLoading fine-tuned model from {CHECKPOINT}...")
    model = MusicGen.get_pretrained('facebook/musicgen-small', device=DEVICE)
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.lm.load_state_dict(ckpt['lm_state_dict'])
    model.lm.eval()
    print(f"  Checkpoint epoch: {ckpt['epoch']}, val_loss: {ckpt['val_loss']:.4f}")

    best_overall_ratio = 0.0
    best_overall_label = None
    best_overall_audio = None

    for item in SHOWCASE_PROMPTS:
        label  = item['label']
        prompt = item['prompt']
        print(f"\n[{label.upper()}]  \"{prompt[:60]}...\"")
        print(f"  Generating {len(PARAM_GRID)} candidates ({DURATION_SHORT}s each)...")

        candidates = []
        for i, params in enumerate(PARAM_GRID):
            audio = generate_candidates(model, prompt, params, DURATION_SHORT, seed=42 + i)
            ratio = harmonic_ratio(audio, SAMPLE_RATE)
            candidates.append((ratio, audio, params))
            print(f"    [{i+1}/{len(PARAM_GRID)}] temp={params['temperature']} "
                  f"cfg={params['cfg_coef']} top_k={params['top_k']}  "
                  f"harmonic_ratio={ratio:.4f}")

        best_ratio, best_audio, best_params = max(candidates, key=lambda x: x[0])
        out_path = SAMPLES_DIR / f"best_{label}.wav"
        sf.write(str(out_path), best_audio, SAMPLE_RATE)
        print(f"  → Saved {out_path}  (harmonic_ratio={best_ratio:.4f}, params={best_params})")

        if best_ratio > best_overall_ratio:
            best_overall_ratio = best_ratio
            best_overall_label = label
            best_overall_audio = (prompt, best_params)

    # ── 30-second showpiece from the most tonal style ────────────────────────
    if best_overall_audio is not None:
        prompt, params = best_overall_audio
        # Use highest cfg_coef for the showpiece
        showcase_params = dict(params, cfg_coef=5.0, temperature=0.85, duration=DURATION_SHOWCASE)
        print(f"\n[SHOWCASE 30s] Best style: {best_overall_label.upper()}")
        print(f"  Generating 30-second showpiece (cfg_coef=5.0, temp=0.85)...")
        torch.manual_seed(7)
        np.random.seed(7)
        model.set_generation_params(
            duration=DURATION_SHOWCASE,
            use_sampling=True,
            top_k=showcase_params['top_k'],
            temperature=showcase_params['temperature'],
            cfg_coef=showcase_params['cfg_coef'],
        )
        with torch.no_grad():
            wav = model.generate([prompt])
        audio_np = wav[0].squeeze(0).cpu().float().numpy()
        out_path = SAMPLES_DIR / 'showcase_30s.wav'
        sf.write(str(out_path), audio_np, SAMPLE_RATE)
        ratio = harmonic_ratio(audio_np, SAMPLE_RATE)
        print(f"  → Saved {out_path}  (harmonic_ratio={ratio:.4f})")

    print(f"\nAll samples saved to {SAMPLES_DIR}/")
    print("Files:")
    for f in sorted(SAMPLES_DIR.glob('*.wav')):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:<30} {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
