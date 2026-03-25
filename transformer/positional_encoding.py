"""
Positional encoding modules for the Transformer model.

This module provides different types of positional encodings:
- Sinusoidal positional encoding (original Transformer paper)
- Learned positional encoding
- Relative positional encoding (optional)
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in the original Transformer paper.

    Uses batch-first format: (batch_size, seq_len, d_model).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Store as (1, max_len, d_model) for batch-first broadcasting
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding using an embedding table."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        x = x + self.pe(positions).unsqueeze(0)
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding that encodes distances between positions."""

    def __init__(self, d_model: int, max_relative_position: int = 32, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_relative_position = max_relative_position
        self.rel_pos_emb = nn.Embedding(2 * max_relative_position + 1, d_model)
        nn.init.normal_(self.rel_pos_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)

        range_vec = torch.arange(seq_len, device=x.device)
        distance_mat = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        distance_mat_clipped = torch.clamp(
            distance_mat,
            -self.max_relative_position,
            self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position

        rel_pos_emb = self.rel_pos_emb(final_mat)  # (seq_len, seq_len, d_model)
        x = x + rel_pos_emb.mean(dim=1, keepdim=True)
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """Factory wrapper supporting different positional encoding types."""

    def __init__(self, d_model: int, max_len: int = 5000,
                 encoding_type: str = "sinusoidal", dropout: float = 0.1):
        super().__init__()

        if encoding_type == "sinusoidal":
            self.encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        elif encoding_type == "learned":
            self.encoding = LearnedPositionalEncoding(d_model, max_len, dropout)
        elif encoding_type == "relative":
            self.encoding = RelativePositionalEncoding(d_model, max_len // 2, dropout)
        else:
            raise ValueError(f"Unknown positional encoding type: {encoding_type}")

        self.encoding_type = encoding_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        return self.encoding(x)

    def get_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.encoding_type == "sinusoidal":
            return self.encoding.pe[:, :seq_len, :].to(device)
        elif self.encoding_type == "learned":
            positions = torch.arange(seq_len, device=device, dtype=torch.long)
            return self.encoding.pe(positions)
        else:
            raise NotImplementedError(f"get_encoding not implemented for {self.encoding_type}")
