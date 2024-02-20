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

    # feed forward network parameters
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
    # Check that the embedding dimension is even as specified in RoFormer paper
    assert head_dim % 2 == 0, "Dimension must be evenly divisible by 2."

    # Build theta parameters
    # theta_i = 10000 ^ (-2(i-1)/dim) for i [1, 2, ..., dim / 2]
    # Shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()

    # Shape: (head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # Construct the positions (m parameter)
    # Shape: (seq_len)
    m = torch.arange(seq_len, device)

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
    # (batch, seq_len, h, head_dim / 2) * (1, seq_len, 1, head_dim / 2) = (batch, seq_len, h, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # (batch, seq_len, h, head_dim / 2) -> (batch, seq_len, h, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (batch, seq_len, h, head_dim / 2, 2) -> (batch, seq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    pass


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
    pass


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
    pass


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
    pass


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
    
# To do:
    # KV Cache
    # Self Attention
    # Feed Forward
    # Encoder
