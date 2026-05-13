# Shakespeare LLM Tokenizer

Production-ready end-to-end example for training BPE, WordPiece, Unigram, and SentencePiece tokenizers from scratch.

## Dataset
Tiny Shakespeare (~10k sentences)

## Features
- Hugging Face `tokenizers` (BPE / WordPiece / Unigram)
- SentencePiece (Unigram & BPE)
- Full import-safe training CLI
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

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Repo created via Elite Agent Agency multi-agent system.