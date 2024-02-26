import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32                   # number of heads for queries
    n_kv_heads: Optional[int] = None    # number of heads for keys and values
    vocab_size: int = -1                # set when tokenizer is loaded

    # Feed forward network parameters
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # KV cache parameters
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))     # gamma parameter
    
    def _norm(self, x: torch.Tensor):
        # (batch, seq_len, dim) * (batch, seq_len, 1) = (batch, seq_len, dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)      # rsqrt = 1 / sqrt(x)

    def forward(self, x: torch.Tensor):
        # (dim) * (batch, seq_len, dim) = (batch, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # Check that the embedding dimension is even as specified in the RoFormer paper
    assert head_dim % 2 == 0, "Dimension must be even (divisible by 2)."

    # Build theta parameters
    # theta_i = 10000 ^ (-2(i-1)/dim) for i [1, 2, ..., dim / 2]
    # Shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()

    # Shape: (head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # Construct the positions (m parameter)
    # Shape: (seq_len)
    m = torch.arange(seq_len, device=device)

    # Multiply each theta by each position using outer product to get a matrix of all m_i * theta_i combinations
    # Shape: (seq_len) outer_product (head_dim / 2) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()

    # Compute complex polar form c = R * exp(i * m * theta), where R = 1
    # Shape: (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (batch, seq_len, h, head_dim) -> (batch, seq_len, h, head_diim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # (batch, seq_len, h, head_dim / 2) * (1, seq_len, 1, head_dim / 2) -> (batch, seq_len, h, head_dim / 2)
    x_rotated = x_complex * freqs_complex

    # (batch, seq_len, h, head_dim / 2) -> (batch, seq_len, h, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)

    # (batch, seq_len, h, head_dim / 2, 2) -> (batch, seq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return(
            # (batch, seq_len, n_kv_heads, 1, head_dim)
            x[:, :, :, None, :]
            # (batch, seq_len, n_kv_heads, n_rep, head_dim)
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            # (batch, seq_len, n_kv_heads * n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        # The number of heads for the keys and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        # The number of heads for the queries
        self.n_heads_q = args.n_heads

        # How many times the heads of the keys and values should be repeated to match the heads of the queries
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # The number of dimensions of each heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape    # (batch, 1, dim)

        # Apply Wq, Wk, and Wv matrices to the queries, keys, and values
        xq = self.wq(x)         # (batch, 1, dim) -> (batch, 1, h_q * head_dim)
        xk = self.wk(x)         # (batch, 1, dim) -> (batch, 1, h_kv * head_dim)
        xv = self.wv(x)         # (batch, 1, dim) -> (batch, 1, h_kv * head_dim)

        # Split matrices by the number of heads
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)    # (batch, 1, h_q * head_dim) -> (batch, 1, h_q, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)   # (batch, 1, h_kv * head_dim) -> (batch, 1, h_kv, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)   # (batch, 1, h_kv * head_dim) -> (batch, 1, h_kv, head_dim)

        # Apply RoPE only to queries and keys; does not change the tensor shapes
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # KV Cache: Replace the entry in the cache with the current token
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # Retrieve all the cached keys and values obtained thus far
        keys = self.cache_k[:batch_size, : start_pos + seq_len]             # (batch, seq_len_kv, h_kw, head_dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]           # (batch, seq_len_kv, h_kw, head_dim)

        # Grouped Multi-Query Self Attention
        # Every group of queries shares the same key and value heads so replicate the KV heads for every Q in the same group
        keys = repeat_kv(keys, self.n_rep)      # (batch, seq_len_kv, h_kv, head_dim) -> (batch, seq_len_kv, h_q, head_dim)
        values = repeat_kv(values, self.n_rep)  # (batch, seq_len_kv, h_kv, head_dim) -> (batch, seq_len_kv, h_q, head_dim)

        # (batch, 1, h_q, head_dim) -> (batch, h_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (batch, h_q, 1, head_dim) @ (batch, h_q, head_dim, seq_len_kv) -> (batch, h_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (batch, h_q, 1, seq_len) @ (batch, h_q, seq_len_kv, head_dim) -> (batch, h_q, 1, head_dim)
        output = torch.matmul(scores, values)

        # (batch, h_q, 1, head_dim) -> (batch, 1, h_q, head_dim) -> (batch, 1, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        # (batch, 1, dim) -> (batch, 1, dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        # E.g. hidden_size = 7, multiple_of = 5 -> (7 + 5 - 1) // 5 = 2 -> 2 * 5 = 10
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (batch, seq_len, dim) -> (batch, seq_len, hidden_dim)
        swish = F.silu(self.w1(x))
        # (batch, seq_len, dim) -> (batch, seq_len, hidden_dim)
        x_V = self.w3(x)
        # (batch, seq_len, hidden_dim) * (batch, seq_len, hidden_dim) -> (batch, seq_len, hidden_dim)
        x = swish * x_V
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, dim)
        x = self.w2(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalize before the self attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # Normalize before the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (batch, seq_len, dim) + (batch, seq_len, dim) -> (batch, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        assert args.vocab_size != -1, "Vocabulary size must be set."

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (batch, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token can be processed at a time."

        # (batch, seq_len) -> (batch, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()

        return output
