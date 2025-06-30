"""
Main Transformer model implementation.

This module contains the complete Transformer architecture with encoder-decoder
structure, including embedding layers, transformer blocks, and output projections.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union

from .attention import MultiHeadAttention, SelfAttention, CrossAttention
from .positional_encoding import PositionalEncoding
from .config import ModelConfig


class FeedForward(nn.Module):
    """
    Feed-forward network used in transformer blocks.
    
    This consists of two linear transformations with a ReLU activation
    in between, and dropout for regularization.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, 
                 activation: str = "relu"):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Input and output dimension
            d_ff: Hidden dimension
            dropout: Dropout probability
            activation: Activation function ("relu", "gelu", "swish")
        """
        super().__init__()
        
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Choose activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "swish":
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the feed-forward layers."""
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.w2(self.dropout(self.activation(self.w1(x))))


class TransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and feed-forward network.
    
    This is the basic building block of the transformer architecture,
    consisting of multi-head self-attention followed by a feed-forward network,
    with residual connections and layer normalization.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6, activation: str = "relu"):
        """
        Initialize transformer block.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
            layer_norm_eps: Epsilon for layer normalization
            activation: Activation function for feed-forward network
        """
        super().__init__()
        
        # Self-attention with residual connection
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Feed-forward network with residual connection
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor or tuple of (output, attention_weights)
        """
        # Self-attention with residual connection
        residual = x
        if return_attention:
            attn_output, attention_weights = self.self_attention(x, x, x, mask, return_attention=True)
        else:
            attn_output = self.self_attention(x, x, x, mask, return_attention=False)
        
        x = self.norm1(residual + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        residual = x
        ff_output = self.feed_forward(x)
        x = self.norm2(residual + self.dropout(ff_output))
        
        if return_attention:
            return x, attention_weights
        else:
            return x


class EncoderBlock(TransformerBlock):
    """
    Encoder block for the transformer encoder.
    
    This is identical to the basic transformer block but specifically
    designed for the encoder part of the transformer.
    """
    pass


class DecoderBlock(nn.Module):
    """
    Decoder block with self-attention, cross-attention, and feed-forward network.
    
    This block includes both self-attention (for the target sequence) and
    cross-attention (to attend to the encoder output).
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6, activation: str = "relu"):
        """
        Initialize decoder block.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
            layer_norm_eps: Epsilon for layer normalization
            activation: Activation function for feed-forward network
        """
        super().__init__()
        
        # Self-attention (causal)
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Cross-attention
        self.cross_attention = CrossAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through decoder block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            encoder_output: Output from encoder
            self_attn_mask: Mask for self-attention (causal)
            cross_attn_mask: Mask for cross-attention
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor or tuple of (output, attention_weights)
        """
        attention_weights = {}
        
        # Self-attention with residual connection
        residual = x
        if return_attention:
            attn_output, self_attn_weights = self.self_attention(x, x, x, self_attn_mask, return_attention=True)
            attention_weights['self_attention'] = self_attn_weights
        else:
            attn_output = self.self_attention(x, x, x, self_attn_mask, return_attention=False)
        
        x = self.norm1(residual + self.dropout(attn_output))
        
        # Cross-attention with residual connection
        residual = x
        if return_attention:
            attn_output, cross_attn_weights = self.cross_attention(x, encoder_output, cross_attn_mask, return_attention=True)
            attention_weights['cross_attention'] = cross_attn_weights
        else:
            attn_output = self.cross_attention(x, encoder_output, cross_attn_mask, return_attention=False)
        
        x = self.norm2(residual + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        residual = x
        ff_output = self.feed_forward(x)
        x = self.norm3(residual + self.dropout(ff_output))
        
        if return_attention:
            return x, attention_weights
        else:
            return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder stack.
    
    This consists of multiple encoder blocks stacked together,
    with optional positional encoding.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize transformer encoder.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        
        # Create encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps
            )
            for _ in range(config.n_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor or tuple of (output, attention_weights_list)
        """
        attention_weights_list = []
        
        for layer in self.layers:
            if return_attention:
                x, attn_weights = layer(x, mask, return_attention=True)
                attention_weights_list.append(attn_weights)
            else:
                x = layer(x, mask, return_attention=False)
        
        x = self.norm(x)
        
        if return_attention:
            return x, attention_weights_list
        else:
            return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder stack.
    
    This consists of multiple decoder blocks stacked together,
    with optional positional encoding.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize transformer decoder.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        
        # Create decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps
            )
            for _ in range(config.n_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Forward pass through decoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            encoder_output: Output from encoder
            self_attn_mask: Mask for self-attention (causal)
            cross_attn_mask: Mask for cross-attention
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor or tuple of (output, attention_weights_list)
        """
        attention_weights_list = []
        
        for layer in self.layers:
            if return_attention:
                x, attn_weights = layer(x, encoder_output, self_attn_mask, cross_attn_mask, return_attention=True)
                attention_weights_list.append(attn_weights)
            else:
                x = layer(x, encoder_output, self_attn_mask, cross_attn_mask, return_attention=False)
        
        x = self.norm(x)
        
        if return_attention:
            return x, attention_weights_list
        else:
            return x


class Transformer(nn.Module):
    """
    Complete Transformer model with encoder-decoder architecture.
    
    This is the main transformer model that combines embeddings,
    positional encoding, encoder, decoder, and output projection.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize transformer model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_len,
            encoding_type=config.pos_encoding_type,
            dropout=config.dropout
        )
        
        # Encoder
        self.encoder = TransformerEncoder(config)
        
        # Decoder
        self.decoder = TransformerDecoder(config)
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights between input embeddings and output projection
        self.output_projection.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize token embeddings
        nn.init.normal_(self.token_embedding.weight, std=self.config.init_std)
        
        # Initialize output projection (already tied to embeddings)
        pass
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
               return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Encode source sequence.
        
        Args:
            src: Source sequence of shape (batch_size, src_len)
            src_mask: Optional mask for source sequence
            return_attention: Whether to return attention weights
            
        Returns:
            Encoder output or tuple of (output, attention_weights)
        """
        # Token embeddings
        src_emb = self.token_embedding(src) * math.sqrt(self.config.d_model)
        
        # Add positional encoding
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        
        # Pass through encoder
        return self.encoder(src_emb, src_mask, return_attention)
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               src_mask: Optional[torch.Tensor] = None,
               return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Decode target sequence.
        
        Args:
            tgt: Target sequence of shape (batch_size, tgt_len)
            encoder_output: Output from encoder
            tgt_mask: Optional mask for target sequence (causal)
            src_mask: Optional mask for source sequence
            return_attention: Whether to return attention weights
            
        Returns:
            Decoder output or tuple of (output, attention_weights)
        """
        # Token embeddings
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.config.d_model)
        
        # Add positional encoding
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        
        # Pass through decoder
        return self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask, return_attention)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass through the complete transformer.
        
        Args:
            src: Source sequence of shape (batch_size, src_len)
            tgt: Target sequence of shape (batch_size, tgt_len)
            src_mask: Optional mask for source sequence
            tgt_mask: Optional mask for target sequence (causal)
            return_attention: Whether to return attention weights
            
        Returns:
            Output logits or tuple of (logits, attention_weights)
        """
        # Encode source sequence
        if return_attention:
            encoder_output, encoder_attention = self.encode(src, src_mask, return_attention=True)
        else:
            encoder_output = self.encode(src, src_mask, return_attention=False)
        
        # Decode target sequence
        if return_attention:
            decoder_output, decoder_attention = self.decode(
                tgt, encoder_output, tgt_mask, src_mask, return_attention=True
            )
        else:
            decoder_output = self.decode(
                tgt, encoder_output, tgt_mask, src_mask, return_attention=False
            )
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        if return_attention:
            attention_weights = {
                'encoder': encoder_attention,
                'decoder': decoder_attention
            }
            return logits, attention_weights
        else:
            return logits
    
    def generate(self, src: torch.Tensor, max_len: int, 
                src_mask: Optional[torch.Tensor] = None,
                temperature: float = 1.0, top_k: Optional[int] = None,
                top_p: Optional[float] = None, pad_token_id: int = 0,
                eos_token_id: int = 1) -> torch.Tensor:
        """
        Generate target sequence using greedy decoding.
        
        Args:
            src: Source sequence
            max_len: Maximum generation length
            src_mask: Optional mask for source sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated sequence
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Encode source sequence
        encoder_output = self.encode(src, src_mask, return_attention=False)
        
        # Initialize target sequence with start token
        tgt = torch.full((batch_size, 1), pad_token_id, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for i in range(max_len - 1):
                # Create causal mask for current sequence
                tgt_mask = self._create_causal_mask(tgt.size(1), device)
                
                # Get predictions
                logits = self.decode(tgt, encoder_output, tgt_mask, src_mask, return_attention=False)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Check if all sequences have ended
                if (next_token == eos_token_id).all():
                    break
        
        return tgt
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for decoder."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size(self) -> str:
        """Get model size in a human-readable format."""
        total_params = self.count_parameters()
        
        if total_params >= 1e9:
            return f"{total_params / 1e9:.2f}B"
        elif total_params >= 1e6:
            return f"{total_params / 1e6:.2f}M"
        elif total_params >= 1e3:
            return f"{total_params / 1e3:.2f}K"
        else:
            return str(total_params) 