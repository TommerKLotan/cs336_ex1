import einx
import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


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
        return einx.dot("... d_in, d_out d_in -> ... d_out", x, self.W)


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


def _init_weights(fan_in: int, fan_out: int, device=None, dtype=None):
    std, cutof = _get_std_and_cutof(fan_in, fan_out)
    return nn.Parameter(
        nn.init.trunc_normal_(torch.zeros([fan_out, fan_in], device=device, dtype=dtype), std=std, a=-cutof, b=cutof)
    )


class SwigluLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        d_ff = d_ff or round(d_model * 8 / 3 / 64) * 64

        self.silu = SiluLayer()
        self.W1 = _init_weights(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = _init_weights(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = _init_weights(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        output = self.silu(einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff"))
        output = output * einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        output = einsum(output, self.W2, "... d_ff, d_model d_ff -> ... d_model")
        return output


class RoPELayer(nn.Module):
    def __init__(self, theta: float | None, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        theta = theta or 1
        theta_powers = torch.pow(theta, torch.arange(0, d_k, 2, device=device) / d_k)
        theta_mat = einx.divide("i, k -> i k", torch.arange(0, max_seq_len, device=device), theta_powers)
        self.register_buffer("cos_mat", torch.cos(theta_mat))
        self.register_buffer("sin_mat", torch.sin(theta_mat))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        pairs = rearrange(x, "... (d two) -> ... d two", two=2)
        x1, x2 = pairs[..., 0], pairs[..., 1]
        cos = self.cos_mat[token_positions].expand_as(x1)
        sin = self.sin_mat[token_positions].expand_as(x1)
        r11 = einx.multiply("... seq d, ... seq d -> ... seq d", x1, cos)
        r12 = einx.multiply("... seq d, ... seq d -> ... seq d", x2, -sin)
        r21 = einx.multiply("... seq d, ... seq d -> ... seq d", x1, sin)
        r22 = einx.multiply("... seq d, ... seq d -> ... seq d", x2, cos)
        y1 = r11 + r12
        y2 = r21 + r22
        output = rearrange([y1, y2], "two ... d -> ... (d two)")
        return output


class SoftmaxLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in_features: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
        max_vals = torch.amax(in_features, dim=dim, keepdim=True)
        exps = torch.exp(in_features - max_vals)

        return exps / torch.sum(exps, dim=dim, keepdim=True)


class ScaledDotProductAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = SoftmaxLayer()

    def forward(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... keys d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:

        d_k = Q.shape[-1]
        dot_prod = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / np.sqrt(d_k)
        mask_values = torch.where(mask, 0, float("-inf"))
        scores = self.softmax(dot_prod + mask_values, dim=-1)
        return einsum(scores, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiheadSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
        theta: float | None = None,
        max_seq_len: int | None = None,
        rope=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        default_d_k = d_model // num_heads
        d_k = d_k or default_d_k
        d_v = d_v or default_d_k

        self.W_O = _init_weights(num_heads * d_v, d_model, device=device, dtype=dtype)

        self.W_QKV = _init_weights(d_model, num_heads * (2 * d_k + d_v), device=device, dtype=dtype)
        if rope is not None:
            self.rope = rope
        elif theta is not None and max_seq_len is not None:
            self.rope = RoPELayer(theta, d_k, max_seq_len, device=device)
        else:
            self.rope = None
        self.attention = ScaledDotProductAttentionLayer()

        self.d_k = d_k
        self.device = device
        self.num_heads = num_heads

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: (Int[Tensor, " ... sequence_length"] | None) = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        seq_len = in_features.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len)
        QKV = einx.dot(
            "... d_model, h_dim_sum d_model -> ... h_dim_sum",
            in_features,
            self.W_QKV,
            h=self.num_heads,
        )
        QK = einx.id(
            "... (two h d_k) -> two h ... d_k", QKV[..., : 2 * self.num_heads * self.d_k], two=2, h=self.num_heads
        )
        Q = QK[0, ...]
        K = QK[1, ...]

        V = einx.id("... (h d_v) -> h ... d_v", QKV[..., 2 * self.num_heads * self.d_k :], h=self.num_heads)

        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)) == 1

        multi_head_attention = self.attention(Q, K, V, attention_mask)
        output = einx.dot("h ... seq_len d_v, d_model (h d_v) -> ... seq_len d_model", multi_head_attention, self.W_O)
        return output


class TransformerBlockLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        rope=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm1 = RMSNormLayer(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNormLayer(d_model, device=device, dtype=dtype)
        self.mha = MultiheadSelfAttentionLayer(
            d_model, num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype
        )
        self.ffn = SwigluLayer(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        output = in_features + self.mha(self.norm1(in_features))
        output = output + self.ffn(self.norm2(output))
        return output


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model, device=device, dtype=dtype)
        rope = RoPELayer(rope_theta, d_model // num_heads, context_length, device)
        self.transformers = nn.Sequential(
            *(
                TransformerBlockLayer(d_model, num_heads, d_ff, rope_theta, context_length, rope, device, dtype)
                for _ in range(num_layers)
            )
        )
        self.norm = RMSNormLayer(d_model, device=device, dtype=dtype)
        self.lin = LinearLayer(d_model, vocab_size, device, dtype)
        self.softmax = SoftmaxLayer()

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        output = self.transformers(self.embedding(in_indices))
        output = self.lin(self.norm(output))
        return output


if __name__ == "__main__":
    d_model = 2
    num_heads = 1

    # in_features = torch.ones(2, d_model)
    in_features = torch.tensor([[1.0, 1], [2, 2]])
    multihead_attention = MultiheadSelfAttentionLayer(d_model, num_heads, theta=None, max_seq_len=None)
    q_proj_weight = torch.ones(d_model, d_model)
    k_proj_weight = 2 * torch.ones(d_model, d_model)
    v_proj_weight = 3 * torch.ones(d_model, d_model)
    o_proj_weight = 4 * torch.eye(d_model)
    W_QKV = rearrange([q_proj_weight, k_proj_weight, v_proj_weight], "num_mats d1 d2 -> (num_mats d1) d2")
    multihead_attention.W_QKV.data = W_QKV
    multihead_attention.W_O.data = o_proj_weight
    output = multihead_attention(in_features)

    print(output)
