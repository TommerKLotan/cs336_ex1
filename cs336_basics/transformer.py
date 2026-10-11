import numpy as np
import torch
from einops import einsum, rearrange
from torch import nn


def _get_std_and_cutof(fan_in, fan_out) -> tuple[float, float]:
    std = np.sqrt(2 / (fan_in + fan_out))
    cutof = 3 * std
    return std, cutof


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std, cutof = _get_std_and_cutof(in_features, out_features)
        self.W = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros([out_features, in_features], device=device, dtype=dtype), std=std, a=-cutof, b=cutof
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        std, cutof = _get_std_and_cutof(num_embeddings, embedding_dim)
        self.embeddings = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros([num_embeddings, embedding_dim], device=device, dtype=dtype), std=std, a=-cutof, b=cutof
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RMSNormLayer(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(einsum(torch.square(x), "... d_model -> ...") / self.d_model + self.eps)
        rms = rearrange(rms, "... -> ... 1")
        result = (x / rms) * self.gamma

        return result.to(in_dtype)


class SiluLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)


class SwigluLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        d_ff = d_ff or round(d_model * 8 / 3 / 64) * 64

        def _init_weights(fan_in: int, fan_out: int):
            std, cutof = _get_std_and_cutof(fan_in, fan_out)
            return nn.Parameter(
                nn.init.trunc_normal_(
                    torch.zeros([fan_out, fan_in], device=device, dtype=dtype), std=std, a=-cutof, b=cutof
                )
            )

        self.silu = SiluLayer()
        self.W1 = _init_weights(d_model, d_ff)
        self.W2 = _init_weights(d_ff, d_model)
        self.W3 = _init_weights(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        output = self.silu(einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff"))
        output = output * einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        output = einsum(output, self.W2, "... d_ff, d_model d_ff -> ... d_model")
        return output
