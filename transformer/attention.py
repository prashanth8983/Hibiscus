"""
Multi-head attention implementation for the Transformer model.

This module provides the core attention mechanism used in transformers,
including scaled dot-product attention, multi-head attention, and
various attention optimizations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism.
    
    This is the core attention mechanism that computes attention weights
    based on the similarity between queries and keys, scaled by the
    square root of the key dimension.
    """
    
    def __init__(self, dropout: float = 0.1):
        """
        Initialize scaled dot-product attention.
        
        Args:
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, n_heads, seq_len, d_k)
            key: Key tensor of shape (batch_size, n_heads, seq_len, d_k)
            value: Value tensor of shape (batch_size, n_heads, seq_len, d_v)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len, seq_len)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    This module splits the input into multiple heads, applies attention
    to each head separately, and then concatenates the results.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 bias: bool = True, attention_type: str = "scaled_dot_product"):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            attention_type: Type of attention mechanism
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # Linear projections for queries, keys, and values
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Attention mechanism
        if attention_type == "scaled_dot_product":
            self.attention = ScaledDotProductAttention(dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the attention layers."""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        
        if self.w_q.bias is not None:
            nn.init.zeros_(self.w_q.bias)
        if self.w_k.bias is not None:
            nn.init.zeros_(self.w_k.bias)
        if self.w_v.bias is not None:
            nn.init.zeros_(self.w_v.bias)
        if self.w_o.bias is not None:
            nn.init.zeros_(self.w_o.bias)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (n_heads, d_k).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor of shape (batch_size, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the heads back into a single dimension.
        
        Args:
            x: Input tensor of shape (batch_size, n_heads, seq_len, d_k)
            
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, n_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, 
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor or tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        q = self._split_heads(self.w_q(query))  # (batch_size, n_heads, seq_len, d_k)
        k = self._split_heads(self.w_k(key))    # (batch_size, n_heads, seq_len, d_k)
        v = self._split_heads(self.w_v(value))  # (batch_size, n_heads, seq_len, d_v)
        
        # Apply attention
        attn_output, attention_weights = self.attention(q, k, v, mask)
        
        # Combine heads
        attn_output = self._combine_heads(attn_output)
        
        # Final linear projection
        output = self.w_o(attn_output)
        
        if return_attention:
            return output, attention_weights
        else:
            return output


class SelfAttention(MultiHeadAttention):
    """
    Self-attention mechanism where query, key, and value come from the same input.
    """
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor or tuple of (output, attention_weights)
        """
        return super().forward(x, x, x, mask, return_attention)


class CrossAttention(MultiHeadAttention):
    """
    Cross-attention mechanism where query comes from one input and key/value from another.
    """
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute cross-attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key_value: Key/value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor or tuple of (output, attention_weights)
        """
        return super().forward(query, key_value, key_value, mask, return_attention)


class AttentionWithResidual(nn.Module):
    """
    Attention mechanism with residual connection and layer normalization.
    
    This is the standard attention block used in transformer layers.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 attention_type: str = "scaled_dot_product"):
        """
        Initialize attention with residual connection.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            dropout: Dropout probability
            attention_type: Type of attention mechanism
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, attention_type=attention_type)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor
            mask: Optional mask tensor
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor or tuple of (output, attention_weights)
        """
        residual = x
        
        # Apply attention
        if return_attention:
            attn_output, attention_weights = self.attention(x, x, x, mask, return_attention=True)
        else:
            attn_output = self.attention(x, x, x, mask, return_attention=False)
        
        # Add residual connection and apply layer norm
        output = self.layer_norm(residual + self.dropout(attn_output))
        
        if return_attention:
            return output, attention_weights
        else:
            return output


def create_attention_mask(seq_len: int, device: torch.device, 
                         causal: bool = True) -> torch.Tensor:
    """
    Create attention mask for causal or non-causal attention.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
        causal: Whether to create causal mask (for decoder)
        
    Returns:
        Attention mask tensor
    """
    if causal:
        # Create causal mask (lower triangular)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
    else:
        # Create full attention mask (no masking)
        mask = torch.zeros(seq_len, seq_len, device=device)
    
    return mask


def create_padding_mask(padding_mask: torch.Tensor, 
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Create padding mask for attention.
    
    Args:
        padding_mask: Boolean tensor indicating padded positions
        attention_mask: Optional existing attention mask
        
    Returns:
        Combined attention mask
    """
    # Create padding mask for attention
    seq_len = padding_mask.size(-1)
    pad_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    pad_mask = pad_mask.expand(-1, -1, seq_len, -1)    # (batch_size, 1, seq_len, seq_len)
    
    if attention_mask is not None:
        # Combine with existing attention mask
        return attention_mask + pad_mask
    else:
        return pad_mask 