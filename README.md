# MetalThroat

Fine-tuning MusicGen to generate Mongolian/Tuvan throat singing (khoomei, sygyt, kargyraa).

Minimal repo contents and quickstart:

- Notebooks: `01_setup_and_inference.ipynb`, `02_data_preparation.ipynb`, `03_finetuning.ipynb`, `04_evaluation.ipynb`
- Dataset manifests: `dataset/train.jsonl`, `dataset/val.jsonl`
- Checkpoints: `checkpoints/` (large; not committed)

Quick start (assumes Python 3.10+ and CUDA-ready PyTorch):

```powershell
python -m pip install -r requirements.txt
# Run setup notebook or open notebooks in JupyterLab
jupyter notebook
```

Notes:
- `T5` and `EnCodec` are kept frozen; only the LM is fine-tuned (see CLAUDE.md).
- Large artifacts (raw audio, processed clips, checkpoints) are ignored by `.gitignore`.
