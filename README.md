---
title: Shakespeare LLM Tokenizer
emoji: 📜
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.50.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Shakespeare LLM Tokenizer

Production-ready end-to-end example for training BPE, WordPiece, Unigram, and SentencePiece tokenizers from scratch.

## Hugging Face Space

This repository includes a small Gradio demo in `app.py` and a committed demo tokenizer under `demo_tokenizer/` so a Space can load without running the full training pipeline. For full artifacts and LM training, use the CLI below.

## Dataset
Tiny Shakespeare (~10k sentences)

## Features
- Hugging Face `tokenizers` (BPE / WordPiece / Unigram)
- SentencePiece (Unigram & BPE)
- Full import-safe training CLI
- PyTorch next-token language model training
- Reproducible pipeline

## Quick Start
```bash
pip install -r requirements.txt
python3 train_tokenizer.py
```

By default this downloads Tiny Shakespeare, writes `tiny_shakespeare_10k.txt`,
and stores tokenizer artifacts in `tokenizer_artifacts/`.

```bash
python3 train_tokenizer.py --help
python3 train_tokenizer.py --reuse-corpus --vocab-size 8000
```

Generated artifacts:

- `huggingface_bpe_tokenizer.json`
- `huggingface_wordpiece_tokenizer.json`
- `huggingface_unigram_tokenizer.json`
- `sentencepiece_unigram.model`
- `sentencepiece_bpe.model`

## Train a Language Model

After training tokenizers, train a compact GRU causal language model on the
tokenized Tiny Shakespeare corpus:

```bash
python3 train_lm.py \
  --corpus-path tiny_shakespeare_10k.txt \
  --tokenizer-path tokenizer_artifacts/huggingface_bpe_tokenizer.json \
  --epochs 3 \
  --batch-size 32
```

The default checkpoint is written to:

```text
model_artifacts/tiny_shakespeare_gru.pt
```

For a quick CPU smoke run:

```bash
python3 train_lm.py --epochs 1 --max-examples 128 --embedding-dim 32 --hidden-dim 64
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Repo created via Elite Agent Agency multi-agent system.
