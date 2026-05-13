import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import requests
import sentencepiece as spm
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

DATASET_URL = "https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt"
DEFAULT_CORPUS_PATH = Path("tiny_shakespeare_10k.txt")
DEFAULT_OUTPUT_DIR = Path("tokenizer_artifacts")
DEFAULT_SENTENCE_LIMIT = 10_000
DEFAULT_MIN_CHARS = 10
DEFAULT_VOCAB_SIZE = 8_000
REQUEST_TIMEOUT_SECONDS = 30
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!)\n])\s+")
SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]


@dataclass(frozen=True)
class TrainingArtifacts:
    corpus: Path
    huggingface_bpe: Path
    huggingface_wordpiece: Path
    huggingface_unigram: Path
    sentencepiece_unigram_model: Path
    sentencepiece_bpe_model: Path


def fetch_dataset(url: str = DATASET_URL) -> str:
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def split_sentences(text: str, limit: int = DEFAULT_SENTENCE_LIMIT, min_chars: int = DEFAULT_MIN_CHARS) -> list[str]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if min_chars < 0:
        raise ValueError("min_chars cannot be negative")

    candidates = SENTENCE_SPLIT_PATTERN.split(text)
    sentences = [sentence.strip() for sentence in candidates if len(sentence.strip()) > min_chars]
    return sentences[:limit]


def write_corpus(sentences: Sequence[str], path: Path) -> Path:
    if not sentences:
        raise ValueError("cannot write an empty corpus")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    return path


def _set_template_processing(tokenizer: Tokenizer) -> None:
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    if cls_id is None or sep_id is None:
        raise RuntimeError("trained tokenizer is missing required special tokens")

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )


def train_huggingface_bpe(corpus_path: Path, output_dir: Path, vocab_size: int = DEFAULT_VOCAB_SIZE) -> Path:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)

    tokenizer.train([str(corpus_path)], trainer)
    _set_template_processing(tokenizer)

    output_path = output_dir / "huggingface_bpe_tokenizer.json"
    tokenizer.save(str(output_path))
    return output_path


def train_huggingface_wordpiece(corpus_path: Path, output_dir: Path, vocab_size: int = DEFAULT_VOCAB_SIZE) -> Path:
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)

    tokenizer.train([str(corpus_path)], trainer)
    _set_template_processing(tokenizer)

    output_path = output_dir / "huggingface_wordpiece_tokenizer.json"
    tokenizer.save(str(output_path))
    return output_path


def train_huggingface_unigram(corpus_path: Path, output_dir: Path, vocab_size: int = DEFAULT_VOCAB_SIZE) -> Path:
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, unk_token="[UNK]")

    tokenizer.train([str(corpus_path)], trainer)
    _set_template_processing(tokenizer)

    output_path = output_dir / "huggingface_unigram_tokenizer.json"
    tokenizer.save(str(output_path))
    return output_path


def train_sentencepiece(
    corpus_path: Path,
    output_dir: Path,
    model_type: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> Path:
    model_prefix = output_dir / f"sentencepiece_{model_type}"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols="[CLS],[SEP],[MASK]",
        hard_vocab_limit=False,
    )
    return model_prefix.with_suffix(".model")


def build_corpus(corpus_path: Path, url: str, limit: int, min_chars: int) -> Path:
    text = fetch_dataset(url)
    sentences = split_sentences(text, limit=limit, min_chars=min_chars)
    return write_corpus(sentences, corpus_path)


def train_all_tokenizers(corpus_path: Path, output_dir: Path, vocab_size: int = DEFAULT_VOCAB_SIZE) -> TrainingArtifacts:
    if not corpus_path.is_file():
        raise FileNotFoundError(f"corpus file does not exist: {corpus_path}")
    if vocab_size < len(SPECIAL_TOKENS):
        raise ValueError(f"vocab_size must be at least {len(SPECIAL_TOKENS)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return TrainingArtifacts(
        corpus=corpus_path,
        huggingface_bpe=train_huggingface_bpe(corpus_path, output_dir, vocab_size),
        huggingface_wordpiece=train_huggingface_wordpiece(corpus_path, output_dir, vocab_size),
        huggingface_unigram=train_huggingface_unigram(corpus_path, output_dir, vocab_size),
        sentencepiece_unigram_model=train_sentencepiece(corpus_path, output_dir, "unigram", vocab_size),
        sentencepiece_bpe_model=train_sentencepiece(corpus_path, output_dir, "bpe", vocab_size),
    )


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser(description="Train BPE, WordPiece, Unigram, and SentencePiece tokenizers.")
    parser.add_argument("--dataset-url", default=DATASET_URL, help="Text dataset URL to download.")
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH, help="Where to write the corpus.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for tokenizer artifacts.")
    parser.add_argument("--limit", type=int, default=DEFAULT_SENTENCE_LIMIT, help="Maximum sentences to keep.")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS, help="Minimum sentence length.")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="Target tokenizer vocab size.")
    parser.add_argument("--reuse-corpus", action="store_true", help="Skip download when --corpus-path already exists.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.reuse_corpus and args.corpus_path.is_file():
        corpus_path = args.corpus_path
        print(f"Reusing corpus: {corpus_path}")
    else:
        print("Fetching Tiny Shakespeare...")
        corpus_path = build_corpus(args.corpus_path, args.dataset_url, args.limit, args.min_chars)
        line_count = len(corpus_path.read_text(encoding="utf-8").splitlines())
        print(f"Loaded {line_count:,} sentence(s) into {corpus_path}")

    artifacts = train_all_tokenizers(corpus_path, args.output_dir, args.vocab_size)
    print("Tokenizer artifacts written:")
    for artifact in artifacts.__dict__.values():
        print(f"- {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())