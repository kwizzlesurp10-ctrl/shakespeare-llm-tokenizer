from __future__ import annotations

from pathlib import Path

import gradio as gr
from tokenizers import Tokenizer

_ROOT = Path(__file__).resolve().parent
DEMO_TOKENIZER_PATH = _ROOT / "demo_tokenizer" / "huggingface_bpe_tokenizer.json"


def load_demo_tokenizer() -> Tokenizer:
    if not DEMO_TOKENIZER_PATH.is_file():
        raise FileNotFoundError(f"Missing demo tokenizer at {DEMO_TOKENIZER_PATH}")
    return Tokenizer.from_file(str(DEMO_TOKENIZER_PATH))


def format_tokenization(text: str) -> str:
    tokenizer = load_demo_tokenizer()
    encoding = tokenizer.encode(text)
    ids_repr = ", ".join(str(i) for i in encoding.ids)
    tokens_repr = ", ".join(encoding.tokens)
    return (
        f"**Token count:** {len(encoding.ids)}\n\n"
        f"**ids:** `{ids_repr}`\n\n"
        f"**tokens:** `{tokens_repr}`"
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Shakespeare LLM Tokenizer") as demo:
        gr.Markdown(
            "# Shakespeare LLM tokenizer\n"
            "Interactive BPE demo using a small tokenizer trained on Tiny Shakespeare "
            "(see `demo_tokenizer/` and `train_tokenizer.py`)."
        )
        text = gr.Textbox(
            label="Input text",
            lines=4,
            placeholder="To be, or not to be, that is the question.",
        )
        output = gr.Markdown()
        text.submit(fn=format_tokenization, inputs=text, outputs=output)
        gr.Button("Tokenize").click(fn=format_tokenization, inputs=text, outputs=output)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
