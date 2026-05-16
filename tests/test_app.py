from __future__ import annotations

import unittest

from tokenizers import Tokenizer

import app as app_module


class TestAppTokenization(unittest.TestCase):
    def test_demo_tokenizer_file_exists(self) -> None:
        self.assertTrue(app_module.DEMO_TOKENIZER_PATH.is_file())

    def test_format_tokenization_returns_markdown(self) -> None:
        result = app_module.format_tokenization("hello world")
        self.assertIn("**ids:**", result)
        self.assertIn("**tokens:**", result)
        self.assertIn("**Token count:**", result)

    def test_load_demo_tokenizer_roundtrip(self) -> None:
        tokenizer = app_module.load_demo_tokenizer()
        self.assertIsInstance(tokenizer, Tokenizer)
        encoding = tokenizer.encode("The king speaks.")
        self.assertGreater(len(encoding.ids), 0)


if __name__ == "__main__":
    unittest.main()
