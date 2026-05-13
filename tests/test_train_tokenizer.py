import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer

import train_tokenizer


class CorpusTests(unittest.TestCase):
    def test_split_sentences_filters_short_entries_and_honors_limit(self) -> None:
        text = "Hi.\nNow is the winter of our discontent.\nToo short.\nTo be, or not to be? That is the question!"

        sentences = train_tokenizer.split_sentences(text, limit=2, min_chars=10)

        self.assertEqual(
            sentences,
            [
                "Now is the winter of our discontent.",
                "To be, or not to be?",
            ],
        )

    def test_split_sentences_rejects_invalid_limits(self) -> None:
        with self.assertRaises(ValueError):
            train_tokenizer.split_sentences("sample", limit=0)

    def test_write_corpus_rejects_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                train_tokenizer.write_corpus([], Path(temp_dir) / "corpus.txt")

    def test_write_corpus_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "corpus.txt"

            result = train_tokenizer.write_corpus(["alpha", "beta"], path)

            self.assertEqual(result, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "alpha\nbeta\n")


class TrainingTests(unittest.TestCase):
    def test_train_all_tokenizers_writes_expected_artifacts(self) -> None:
        corpus = [
            "Now is the winter of our discontent made glorious summer by this sun of York.",
            "To be, or not to be, that is the question.",
            "All the world's a stage, and all the men and women merely players.",
            "Some are born great, some achieve greatness, and some have greatness thrust upon them.",
            "Cowards die many times before their deaths; the valiant never taste of death but once.",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus_path = train_tokenizer.write_corpus(corpus, root / "corpus.txt")
            artifacts = train_tokenizer.train_all_tokenizers(corpus_path, root / "out", vocab_size=64)

            expected_paths = [
                artifacts.huggingface_bpe,
                artifacts.huggingface_wordpiece,
                artifacts.huggingface_unigram,
                artifacts.sentencepiece_unigram_model,
                artifacts.sentencepiece_bpe_model,
            ]
            for path in expected_paths:
                self.assertTrue(path.is_file(), f"missing artifact: {path}")

            for tokenizer_path in [
                artifacts.huggingface_bpe,
                artifacts.huggingface_wordpiece,
                artifacts.huggingface_unigram,
            ]:
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                self.assertIsNotNone(tokenizer.token_to_id("[UNK]"))
                self.assertIsNotNone(tokenizer.token_to_id("[CLS]"))
                self.assertGreater(tokenizer.get_vocab_size(), 0)


if __name__ == "__main__":
    unittest.main()
