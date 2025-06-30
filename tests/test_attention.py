"""
Unit tests for attention mechanisms.

This module contains tests for the attention components including
scaled dot-product attention, multi-head attention, and attention masks.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

# Add the parent directory to the path to import the transformer package
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from transformer.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    create_attention_mask,
    create_padding_mask
)


class TestScaledDotProductAttention(unittest.TestCase):
    """Test cases for ScaledDotProductAttention."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.attention = ScaledDotProductAttention(dropout=0.1)
        self.batch_size = 2
        self.n_heads = 4
        self.seq_len = 10
        self.d_k = 8
        self.d_v = 8
    
    def test_attention_forward(self):
        """Test forward pass of scaled dot-product attention."""
        # Create sample inputs
        query = torch.randn(self.batch_size, self.n_heads, self.seq_len, self.d_k)
        key = torch.randn(self.batch_size, self.n_heads, self.seq_len, self.d_k)
        value = torch.randn(self.batch_size, self.n_heads, self.seq_len, self.d_v)
        
        # Temporarily disable dropout for testing
        original_dropout = self.attention.dropout.p
        self.attention.dropout.p = 0.0
        
        # Forward pass
        output, attention_weights = self.attention(query, key, value)
        
        # Restore dropout
        self.attention.dropout.p = original_dropout
        
        # Check output shapes
        self.assertEqual(output.shape, (self.batch_size, self.n_heads, self.seq_len, self.d_v))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.n_heads, self.seq_len, self.seq_len))
        
        # Check that attention weights sum to 1 for each row
        attention_sums = attention_weights.sum(dim=-1)
        torch.testing.assert_close(attention_sums, torch.ones_like(attention_sums), rtol=1e-5, atol=1e-6)
    
    def test_attention_with_mask(self):
        """Test attention with mask."""
        # Create sample inputs
        query = torch.randn(self.batch_size, self.n_heads, self.seq_len, self.d_k)
        key = torch.randn(self.batch_size, self.n_heads, self.seq_len, self.d_k)
        value = torch.randn(self.batch_size, self.n_heads, self.seq_len, self.d_v)
        
        # Create mask (mask out last 2 positions)
        mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)
        mask[:, :, :, -2:] = 0
        
        # Forward pass with mask
        output, attention_weights = self.attention(query, key, value, mask)
        
        # Check that masked positions have zero attention weights
        masked_weights = attention_weights[:, :, :, -2:]
        torch.testing.assert_close(masked_weights, torch.zeros_like(masked_weights), rtol=1e-5, atol=1e-6)


class TestMultiHeadAttention(unittest.TestCase):
    """Test cases for MultiHeadAttention."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 64
        self.n_heads = 4
        self.seq_len = 10
        self.batch_size = 2
        self.attention = MultiHeadAttention(self.d_model, self.n_heads, dropout=0.1)
    
    def test_attention_creation(self):
        """Test multi-head attention creation."""
        self.assertEqual(self.attention.d_model, self.d_model)
        self.assertEqual(self.attention.n_heads, self.n_heads)
        self.assertEqual(self.attention.d_k, self.d_model // self.n_heads)
        self.assertEqual(self.attention.d_v, self.d_model // self.n_heads)
    
    def test_attention_forward(self):
        """Test forward pass of multi-head attention."""
        # Create sample inputs
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = self.attention(query, key, value)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_attention_with_return_weights(self):
        """Test attention with return_attention=True."""
        # Create sample inputs
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass with attention weights
        output, attention_weights = self.attention(query, key, value, return_attention=True)
        
        # Check shapes
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.n_heads, self.seq_len, self.seq_len))
    
    def test_invalid_d_model_divisible_by_n_heads(self):
        """Test that d_model must be divisible by n_heads."""
        with self.assertRaises(AssertionError):
            MultiHeadAttention(d_model=65, n_heads=4)  # 65 not divisible by 4


class TestSelfAttention(unittest.TestCase):
    """Test cases for SelfAttention."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 64
        self.n_heads = 4
        self.seq_len = 10
        self.batch_size = 2
        self.attention = SelfAttention(self.d_model, self.n_heads, dropout=0.1)
    
    def test_self_attention_forward(self):
        """Test forward pass of self-attention."""
        # Create sample input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = self.attention(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_self_attention_with_mask(self):
        """Test self-attention with mask."""
        # Create sample input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Create mask
        mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)
        mask[:, :, :, -2:] = 0  # Mask out last 2 positions
        
        # Forward pass with mask
        output = self.attention(x, mask)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))


class TestCrossAttention(unittest.TestCase):
    """Test cases for CrossAttention."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 64
        self.n_heads = 4
        self.seq_len = 10
        self.batch_size = 2
        self.attention = CrossAttention(self.d_model, self.n_heads, dropout=0.1)
    
    def test_cross_attention_forward(self):
        """Test forward pass of cross-attention."""
        # Create sample inputs
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key_value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = self.attention(query, key_value)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_cross_attention_with_mask(self):
        """Test cross-attention with mask."""
        # Create sample inputs
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key_value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Create mask
        mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)
        mask[:, :, :, -2:] = 0  # Mask out last 2 positions
        
        # Forward pass with mask
        output = self.attention(query, key_value, mask)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))


class TestAttentionMasks(unittest.TestCase):
    """Test cases for attention mask utilities."""
    
    def test_create_attention_mask(self):
        """Test creating attention masks."""
        seq_len = 10
        device = torch.device('cpu')
        
        # Test causal mask
        causal_mask = create_attention_mask(seq_len, device, causal=True)
        self.assertEqual(causal_mask.shape, (seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    self.assertEqual(causal_mask[i, j].item(), float('-inf'))
                else:
                    self.assertEqual(causal_mask[i, j].item(), 0.0)
        
        # Test non-causal mask
        non_causal_mask = create_attention_mask(seq_len, device, causal=False)
        self.assertEqual(non_causal_mask.shape, (seq_len, seq_len))
        self.assertTrue(torch.all(non_causal_mask == 0.0))
    
    def test_create_padding_mask(self):
        """Test creating padding masks."""
        batch_size = 2
        seq_len = 10
        
        # Create padding mask (last 2 positions are padding)
        padding_mask = torch.ones(batch_size, seq_len)
        padding_mask[:, -2:] = 0
        
        # Create attention mask from padding mask
        attention_mask = create_padding_mask(padding_mask)
        self.assertEqual(attention_mask.shape, (batch_size, 1, seq_len, seq_len))
        
        # Check that attention is allowed (mask=1) only when both query and key positions are valid
        # and masked (mask=0) when either query or key position is padding
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    if padding_mask[b, i] == 1 and padding_mask[b, j] == 1:
                        # Both positions are valid, so attention is allowed
                        self.assertEqual(attention_mask[b, 0, i, j].item(), 1.0)
                    else:
                        # At least one position is padding, so attention is masked
                        self.assertEqual(attention_mask[b, 0, i, j].item(), 0.0)


if __name__ == '__main__':
    unittest.main() 