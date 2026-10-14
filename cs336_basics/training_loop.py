import json
import secrets
import string
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import typer
from tokenization import MAIN_DIR, TOKENIZATION_MODELS_DIR, TOKENIZED_DATA_DIR, Tokenizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cs336_basics.training import (
    AdamW,
    cross_entropy,
    get_batch,
    load_checkpoint,
    perplexity_from_loss,
    save_checkpoint,
)
from cs336_basics.transformer import TransformerLM

MODELS_DIR = MAIN_DIR / "models"
CONFIG_FILE_NAME = "config.json"
LOG_DIR = MAIN_DIR / "logs"


def generate_model_uid(length=10):
    # Combines uppercase letters, lowercase letters, and digits
    characters = string.ascii_letters + string.digits
    return "".join(secrets.choice(characters) for _ in range(length))


def create_model_checkpoint(model_id, model, optimizer, iteration):
    save_checkpoint(model, optimizer, iteration, MODELS_DIR / model_id / f"chkp_{iteration}")


def get_vocab_size(tokenization_model_name: str = None):
    tokenizer = Tokenizer.from_files(TOKENIZATION_MODELS_DIR / tokenization_model_name)
    return len(tokenizer.vocab)


def load_datasets(dataset_name: str):
    dataset_file_names = [TOKENIZED_DATA_DIR / f"{dataset_name}-{x}.npy" for x in ["train", "valid"]]
    return tuple(np.load(file_name, mmap_mode="r") for file_name in dataset_file_names)


def train_loop(
    model_id: str,
    config: dict,
    model: TransformerLM,
    optimizer: AdamW,
    train_dataset: npt.NDArray,
    valid_dataset: npt.NDArray,
    iter_num: int = 0,
):
    print(f"Training model: {model_id} from iter {iter_num}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_dir = LOG_DIR / model_id
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    train_config = config["training"]
    model_config = config["model"]
    for iter_num in tqdm(range(iter_num, train_config["num_iters"])):
        # Training step
        optimizer.zero_grad()
        model.train()
        train_batch, train_batch_target = get_batch(
            train_dataset, train_config["train_batch_size"], model_config["context_length"], device
        )
        logits = model(train_batch)
        loss = cross_entropy(logits, train_batch_target)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.inference_mode():
            valid_batch, valid_batch_target = get_batch(
                valid_dataset, train_config["valid_batch_size"], model_config["context_length"], device
            )
            valid_loss = cross_entropy(model(valid_batch), valid_batch_target)
            valid_perplexity = perplexity_from_loss(valid_loss)

        # Logging
        writer.add_scalar("Loss/train", loss.item(), iter_num + 1)
        writer.add_scalar("Loss/validation", valid_loss.item(), iter_num + 1)
        writer.add_scalar("Perplexity/validation", valid_perplexity.item(), iter_num + 1)

        # Checkpoint
        if iter_num + 1 % train_config["iters_between_checkpoints"] == 0 or iter_num == train_config["num_iters"] - 1:
            create_model_checkpoint(
                model_id,
                model,
                optimizer,
                iter_num + 1,
            )
    writer.close()


def train_loop_wrapper(
    resume: str = None,
    just_generate_config: bool = False,
    tokenization_model_name: str = "TinyStoriesV2-GPT4-train-bpe-2026-10-08_19-11-45",
    dataset_name: str = "TinyStoriesV2-GPT4-train-bpe-2026-10-08_19-11-45-TinyStoriesV2-GPT4",
    context_length: int = 16,
    d_model: int = 64,
    num_layers: int = 3,
    num_heads: int = 4,
    d_ff: int = 128,
    rope_theta: float = 10_000,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    num_iters: int = 1000,
    train_batch_size: int = 32,
    valid_batch_size: int = 32,
    iters_between_checkpoints: int = 10,
):
    if resume is None:
        model_id = generate_model_uid()

    else:
        model_id = resume

    model_dir = MODELS_DIR / model_id
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    config_file_path = model_dir / CONFIG_FILE_NAME
    if resume is None:
        config = {
            "tokenization_model_name": tokenization_model_name,
            "model": {
                "vocab_size": get_vocab_size(tokenization_model_name),
                "context_length": context_length,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "d_ff": d_ff,
                "rope_theta": rope_theta,
            },
            "optimizer": {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps},
            "training": {
                "num_iters": num_iters,
                "train_batch_size": train_batch_size,
                "valid_batch_size": valid_batch_size,
                "iters_between_checkpoints": iters_between_checkpoints,
            },
        }
        config_file_path.write_text(json.dumps(config))

    else:
        config = json.loads(config_file_path.read_text())

    if just_generate_config:
        return

    model = TransformerLM(**config["model"])
    optimizer = AdamW(model.parameters(), **config["optimizer"])
    iter_num = 0
    if resume is not None:
        chkp_file_name = max(x for x in model_dir.iterdir() if "chkp" in str(x))
        chkp_file_path = model_dir / chkp_file_name
        iter_num = load_checkpoint(chkp_file_path, model, optimizer)

    train_dataset, valid_dataset = load_datasets(dataset_name)
    train_loop(model_id, config, model, optimizer, train_dataset, valid_dataset, iter_num)


if __name__ == "__main__":
    typer.run(train_loop_wrapper)
