import os
from collections.abc import Callable, Iterable
from typing import IO, BinaryIO

import einx
import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(logits: Float[Tensor, "... d"], target: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    logits = einx.id("... d -> (...) d", logits)
    logits = logits - torch.amax(logits, dim=-1, keepdim=True)
    exp_sum = torch.sum(torch.exp(logits), dim=-1)
    losses = torch.log(exp_sum) - logits[torch.arange(logits.size(0)), target]
    return torch.mean(losses)


def perplexity_from_loss(losses: Float[Tensor, "..."]) -> Float[Tensor, ""]:
    return torch.exp(einx.mean("... -> 1", losses))


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                beta1, beta2 = group["betas"]
                weight_decay = group["weight_decay"]
                lr = group["lr"]
                eps = group["eps"]

                g = p.grad.data
                lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr * weight_decay * p.data
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g**2
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def cosine_learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate + 0.5 * (
            1 + np.cos(np.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        ) * (max_learning_rate - min_learning_rate)


def do_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6):
    def gen_grads():
        return (p.grad.data for p in parameters if p.grad is not None)

    norm = np.sqrt(sum(torch.sum(g**2).item() for g in gen_grads()))
    if norm > max_l2_norm:
        for g in gen_grads():
            g *= max_l2_norm / (norm + eps)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    dataset_tensor = torch.from_numpy(dataset)
    max_start = len(dataset) - context_length - 1
    starts = torch.randint(0, max_start + 1, (batch_size, 1))
    indices = starts + torch.arange(context_length)

    batches = dataset_tensor[indices].to(device)
    targets = dataset_tensor[indices + 1].to(device)
    return batches, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint_dict = dict(
        iteration=iteration,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    torch.save(checkpoint_dict, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
) -> int:
    checkpoint_dict = torch.load(src)
    if model is not None:
        model.load_state_dict(checkpoint_dict["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    return checkpoint_dict["iteration"]
