# AGENTS.md

## Cursor Cloud specific instructions

This is a Python ML pipeline (not a web app) for training NLP tokenizers and a GRU language model on Tiny Shakespeare. No databases, Docker, or external services are required.

### Running commands

- **Tests:** `python3 -m unittest discover -s tests -v` (9 tests, ~2s, all self-contained with temp dirs)
- **Train tokenizers:** `python3 train_tokenizer.py` (downloads corpus from GitHub on first run; use `--reuse-corpus` to skip)
- **Train LM (smoke):** `python3 train_lm.py --epochs 1 --max-examples 128 --embedding-dim 32 --hidden-dim 64`
- **Full LM training:** see `README.md` for default flags

### Gotchas

- `train_lm.py` requires tokenizer artifacts to exist first — always run `train_tokenizer.py` before `train_lm.py`.
- Generated artifacts (`tokenizer_artifacts/`, `model_artifacts/`, `tiny_shakespeare_10k.txt`) are git-ignored.
- No linter is configured in the repo; there is no `pyproject.toml`, `setup.cfg`, or linter config file.
- PyTorch is installed as CPU-only by default on this VM; `--device cpu` is the default and works fine.
