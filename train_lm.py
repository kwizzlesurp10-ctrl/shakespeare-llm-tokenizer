from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from tokenizers import Tokenizer
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_CORPUS_PATH = Path("tiny_shakespeare_10k.txt")
DEFAULT_TOKENIZER_PATH = Path("tokenizer_artifacts/huggingface_bpe_tokenizer.json")
DEFAULT_OUTPUT_DIR = Path("model_artifacts")
DEFAULT_CHECKPOINT_NAME = "tiny_shakespeare_gru.pt"
DEFAULT_BLOCK_SIZE = 64
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_HIDDEN_DIM = 256
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_SEED = 13


@dataclass(frozen=True)
class LanguageModelConfig:
    vocab_size: int
    block_size: int = DEFAULT_BLOCK_SIZE
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM


@dataclass(frozen=True)
class TrainingConfig:
    corpus_path: Path = DEFAULT_CORPUS_PATH
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    block_size: int = DEFAULT_BLOCK_SIZE
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_examples: int | None = None
    seed: int = DEFAULT_SEED
    device: str = "cpu"


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    losses: list[float]
    sample: str


class TokenBlockDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, token_ids: Sequence[int], block_size: int, max_examples: int | None = None) -> None:
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        if len(token_ids) <= block_size:
            raise ValueError("token_ids must contain more tokens than block_size")

        example_count = len(token_ids) - block_size
        if max_examples is not None:
            if max_examples < 1:
                raise ValueError("max_examples must be at least 1 when provided")
            example_count = min(example_count, max_examples)

        self._token_ids = list(token_ids)
        self._block_size = block_size
        self._example_count = example_count

    def __len__(self) -> int:
        return self._example_count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0 or index >= self._example_count:
            raise IndexError(index)

        window = self._token_ids[index : index + self._block_size + 1]
        inputs = torch.tensor(window[:-1], dtype=torch.long)
        targets = torch.tensor(window[1:], dtype=torch.long)
        return inputs, targets


class TinyCausalLanguageModel(nn.Module):
    def __init__(self, config: LanguageModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.recurrent = nn.GRU(config.embedding_dim, config.hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        embedded = self.embedding(input_ids)
        hidden_states, _ = self.recurrent(embedded)
        return self.output(self.norm(hidden_states))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(path: Path) -> Tokenizer:
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer file does not exist: {path}")
    return Tokenizer.from_file(str(path))


def load_token_ids(corpus_path: Path, tokenizer: Tokenizer) -> list[int]:
    if not corpus_path.is_file():
        raise FileNotFoundError(f"corpus file does not exist: {corpus_path}")

    text = corpus_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("corpus file is empty")

    token_ids = tokenizer.encode(text).ids
    if not token_ids:
        raise ValueError("tokenizer produced no token ids")
    return token_ids


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device)


def train_one_epoch(
    model: TinyCausalLanguageModel,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    batch_count = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        batch_count += 1

    if batch_count == 0:
        raise RuntimeError("no batches were produced for training")
    return total_loss / batch_count


def generate_text(
    model: TinyCausalLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 40,
) -> str:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    model.eval()
    token_ids = tokenizer.encode(prompt).ids
    if not token_ids:
        token_ids = [0]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = token_ids[-model.config.block_size :]
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_id = int(torch.argmax(logits[0, -1], dim=-1).detach().cpu())
            token_ids.append(next_id)

    return tokenizer.decode(token_ids, skip_special_tokens=True)


def save_checkpoint(
    model: TinyCausalLanguageModel,
    config: TrainingConfig,
    model_config: LanguageModelConfig,
    losses: Sequence[float],
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / DEFAULT_CHECKPOINT_NAME
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": {
                **asdict(config),
                "corpus_path": str(config.corpus_path),
                "tokenizer_path": str(config.tokenizer_path),
                "output_dir": str(config.output_dir),
            },
            "losses": list(losses),
        },
        checkpoint_path,
    )
    return checkpoint_path


def train_language_model(config: TrainingConfig, sample_prompt: str = "To be") -> TrainingResult:
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    set_seed(config.seed)
    device = resolve_device(config.device)
    tokenizer = load_tokenizer(config.tokenizer_path)
    token_ids = load_token_ids(config.corpus_path, tokenizer)
    dataset = TokenBlockDataset(token_ids, block_size=config.block_size, max_examples=config.max_examples)
    dataloader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    model_config = LanguageModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=config.block_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )
    model = TinyCausalLanguageModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses = [train_one_epoch(model, dataloader, optimizer, device) for _ in range(config.epochs)]
    sample = generate_text(model, tokenizer, sample_prompt, device)
    checkpoint_path = save_checkpoint(model, config, model_config, losses)
    return TrainingResult(checkpoint_path=checkpoint_path, losses=losses, sample=sample)


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser(description="Train a compact next-token language model on Tiny Shakespeare.")
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sample-prompt", default="To be")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = train_language_model(
        TrainingConfig(
            corpus_path=args.corpus_path,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            block_size=args.block_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_examples=args.max_examples,
            seed=args.seed,
            device=args.device,
        ),
        sample_prompt=args.sample_prompt,
    )

    print(f"Checkpoint written: {result.checkpoint_path}")
    print("Losses:")
    for epoch_index, loss in enumerate(result.losses, start=1):
        print(f"- epoch {epoch_index}: {loss:.4f}")
    print("Sample:")
    print(result.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
