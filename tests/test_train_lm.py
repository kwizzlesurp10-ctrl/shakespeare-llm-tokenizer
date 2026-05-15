import tempfile
import unittest
from pathlib import Path

import torch

import train_lm
import train_tokenizer


class TokenBlockDatasetTests(unittest.TestCase):
    def test_dataset_returns_shifted_input_target_windows(self) -> None:
        dataset = train_lm.TokenBlockDataset([1, 2, 3, 4, 5], block_size=3)

        inputs, targets = dataset[0]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(inputs.tolist(), [1, 2, 3])
        self.assertEqual(targets.tolist(), [2, 3, 4])

    def test_dataset_validates_window_size(self) -> None:
        with self.assertRaises(ValueError):
            train_lm.TokenBlockDataset([1, 2, 3], block_size=1)


class TinyCausalLanguageModelTests(unittest.TestCase):
    def test_forward_returns_vocab_logits_for_each_token(self) -> None:
        model = train_lm.TinyCausalLanguageModel(
            train_lm.LanguageModelConfig(vocab_size=17, block_size=4, embedding_dim=8, hidden_dim=12)
        )

        logits = model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))

        self.assertEqual(tuple(logits.shape), (1, 4, 17))


class TrainLanguageModelTests(unittest.TestCase):
    def test_train_language_model_writes_checkpoint(self) -> None:
        corpus = [
            "To be, or not to be, that is the question.",
            "Now is the winter of our discontent.",
            "All the world's a stage, and all the men and women merely players.",
            "The course of true love never did run smooth.",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus_path = train_tokenizer.write_corpus(corpus * 4, root / "corpus.txt")
            tokenizer_path = train_tokenizer.train_huggingface_bpe(corpus_path, root / "tokenizers", vocab_size=64)

            result = train_lm.train_language_model(
                train_lm.TrainingConfig(
                    corpus_path=corpus_path,
                    tokenizer_path=tokenizer_path,
                    output_dir=root / "models",
                    block_size=8,
                    embedding_dim=8,
                    hidden_dim=16,
                    batch_size=4,
                    epochs=1,
                    learning_rate=1e-3,
                    max_examples=8,
                    seed=7,
                    device="cpu",
                )
            )

            self.assertTrue(result.checkpoint_path.is_file())
            self.assertEqual(len(result.losses), 1)
            self.assertGreater(result.losses[0], 0)
            self.assertTrue(result.sample)


if __name__ == "__main__":
    unittest.main()
