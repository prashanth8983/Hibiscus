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
import torch.nn.functional as F
from typing import Optional, Tuple


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in the original Transformer paper.
    
    This encoding allows the model to learn to attend by relative positions,
    since for any fixed offset k, PE(pos+k) can be represented as a linear
    function of PE(pos).
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize sinusoidal positional encoding.
        
        Args:
            d_model: The dimension of the model embeddings
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create division term for different frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding that learns position embeddings from scratch.
    
    This is often used in modern transformer implementations as it can
    potentially learn better position representations than the fixed
    sinusoidal encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize learned positional encoding.
        
        Args:
            d_model: The dimension of the model embeddings
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        
        # Initialize with small random values
        nn.init.normal_(self.pe.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(0)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        pos_encoding = self.pe(positions).unsqueeze(1)  # (seq_len, 1, d_model)
        x = x + pos_encoding
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding that encodes relative distances between positions.
    
    This is used in some modern transformer variants to better handle
    long sequences and relative position information.
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 32, dropout: float = 0.1):
        """
        Initialize relative positional encoding.
        
        Args:
            d_model: The dimension of the model embeddings
            max_relative_position: Maximum relative position to encode
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_relative_position = max_relative_position
        
        # Create relative position embeddings
        self.rel_pos_emb = nn.Embedding(2 * max_relative_position + 1, d_model)
        nn.init.normal_(self.rel_pos_emb.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add relative positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with relative positional encoding added
        """
        seq_len = x.size(0)
        
        # Create relative position indices
        range_vec = torch.arange(seq_len, device=x.device)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to non-negative indices
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # Get relative position embeddings
        rel_pos_emb = self.rel_pos_emb(final_mat)  # (seq_len, seq_len, d_model)
        
        # Add to input (this is a simplified version)
        x = x + rel_pos_emb.mean(dim=1, keepdim=True)
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """
    Main positional encoding class that supports different encoding types.
    
    This is the main interface for positional encoding in the transformer.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, 
                 encoding_type: str = "sinusoidal", dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: The dimension of the model embeddings
            max_len: Maximum sequence length
            encoding_type: Type of positional encoding ("sinusoidal", "learned", "relative")
            dropout: Dropout probability
        """
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
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return self.encoding(x)
    
    def get_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get positional encoding for a given sequence length.
        
        Args:
            seq_len: Sequence length
            device: Device to create tensor on
            
        Returns:
            Positional encoding tensor
        """
        if self.encoding_type == "sinusoidal":
            return self.encoding.pe[:seq_len, :].to(device)
        elif self.encoding_type == "learned":
            positions = torch.arange(seq_len, device=device, dtype=torch.long)
            return self.encoding.pe(positions)
        else:
            raise NotImplementedError(f"get_encoding not implemented for {self.encoding_type}")


def create_positional_encoding(d_model: int, max_len: int = 5000,
                             encoding_type: str = "sinusoidal", 
                             dropout: float = 0.1) -> PositionalEncoding:
    """
    Factory function to create positional encoding.
    
    Args:
        d_model: The dimension of the model embeddings
        max_len: Maximum sequence length
        encoding_type: Type of positional encoding
        dropout: Dropout probability
        
    Returns:
        PositionalEncoding instance
    """
    return PositionalEncoding(d_model, max_len, encoding_type, dropout) 