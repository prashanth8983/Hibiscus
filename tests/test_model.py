"""
Unit tests for the Transformer model.

This module contains tests for the core transformer functionality
including model creation, forward pass, and basic operations.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

# Add the parent directory to the path to import the transformer package
sys.path.append(str(Path(__file__).parent.parent))

from transformer import Transformer, ModelConfig, Tokenizer


class TestModelConfig(unittest.TestCase):
    """Test cases for ModelConfig."""
    
    def test_valid_config(self):
        """Test creating a valid model configuration."""
        config = ModelConfig(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            max_seq_len=64,
            dropout=0.1
        )
        
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.d_model, 128)
        self.assertEqual(config.n_heads, 4)
        self.assertEqual(config.n_layers, 2)
        self.assertEqual(config.d_ff, 512)
        self.assertEqual(config.max_seq_len, 64)
        self.assertEqual(config.dropout, 0.1)
    
    def test_invalid_d_model_divisible_by_n_heads(self):
        """Test that d_model must be divisible by n_heads."""
        with self.assertRaises(ValueError):
            ModelConfig(
                vocab_size=1000,
                d_model=127,  # Not divisible by 4
                n_heads=4,
                n_layers=2,
                d_ff=512
            )
    
    def test_invalid_vocab_size(self):
        """Test that vocab_size must be positive."""
        with self.assertRaises(ValueError):
            ModelConfig(
                vocab_size=0,  # Invalid
                d_model=128,
                n_heads=4,
                n_layers=2,
                d_ff=512
            )
    
    def test_invalid_dropout(self):
        """Test that dropout must be between 0 and 1."""
        with self.assertRaises(ValueError):
            ModelConfig(
                vocab_size=1000,
                d_model=128,
                n_heads=4,
                n_layers=2,
                d_ff=512,
                dropout=1.5  # Invalid
            )


class TestTokenizer(unittest.TestCase):
    """Test cases for Tokenizer."""
    
    def test_word_tokenizer(self):
        """Test word tokenizer functionality."""
        tokenizer = Tokenizer(
            tokenizer_type="word",
            vocab_size=100,
            min_freq=1
        )
        
        # Train on sample data
        texts = ["hello world", "hello there", "world is great"]
        tokenizer.train(texts)
        
        # Test tokenization
        tokens = tokenizer.tokenize("hello world")
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
        
        # Test encoding
        token_ids = tokenizer.encode("hello world")
        self.assertIsInstance(token_ids, list)
        self.assertTrue(len(token_ids) > 0)
        
        # Test decoding
        decoded = tokenizer.decode(token_ids)
        self.assertIsInstance(decoded, str)
    
    def test_character_tokenizer(self):
        """Test character tokenizer functionality."""
        tokenizer = Tokenizer(
            tokenizer_type="char",
            vocab_size=100,
            min_freq=1
        )
        
        # Train on sample data
        texts = ["hello", "world", "test"]
        tokenizer.train(texts)
        
        # Test tokenization
        tokens = tokenizer.tokenize("hello")
        self.assertEqual(tokens, list("hello"))
        
        # Test encoding and decoding
        token_ids = tokenizer.encode("hello")
        decoded = tokenizer.decode(token_ids)
        self.assertIsInstance(decoded, str)


class TestTransformer(unittest.TestCase):
    """Test cases for Transformer model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=32,
            dropout=0.1
        )
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_model_creation(self, mock_cuda):
        """Test creating a transformer model."""
        model = Transformer(self.config)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.config, self.config)
        
        # Test parameter count
        param_count = model.count_parameters()
        self.assertGreater(param_count, 0)
        
        # Test model size
        model_size = model.get_model_size()
        self.assertIsInstance(model_size, str)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_model_forward_pass(self, mock_cuda):
        """Test model forward pass."""
        import torch
        
        model = Transformer(self.config)
        model.eval()
        
        # Create sample input
        batch_size, seq_len = 2, 10
        src_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        tgt_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        with torch.no_grad():
            outputs = model(src_ids, tgt_ids)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, self.config.vocab_size)
        self.assertEqual(outputs.shape, expected_shape)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_model_encode_decode(self, mock_cuda):
        """Test model encode and decode methods."""
        import torch
        
        model = Transformer(self.config)
        model.eval()
        
        # Create sample input
        batch_size, seq_len = 2, 10
        src_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        # Test encode
        with torch.no_grad():
            encoder_output = model.encode(src_ids)
        
        self.assertIsInstance(encoder_output, torch.Tensor)
        self.assertEqual(encoder_output.shape[0], batch_size)
        self.assertEqual(encoder_output.shape[2], self.config.d_model)
        
        # Test decode
        tgt_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            decoder_output = model.decode(tgt_ids, encoder_output)
        
        self.assertIsInstance(decoder_output, torch.Tensor)
        self.assertEqual(decoder_output.shape[0], batch_size)
        self.assertEqual(decoder_output.shape[2], self.config.d_model)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_tokenizer_to_model_pipeline(self):
        """Test the complete pipeline from tokenizer to model."""
        # Create tokenizer
        tokenizer = Tokenizer(
            tokenizer_type="word",
            vocab_size=100,
            min_freq=1
        )
        
        # Train tokenizer
        texts = ["hello world", "machine learning", "deep learning"]
        tokenizer.train(texts)
        
        # Create model config
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=32,
            dropout=0.1
        )
        
        # Create model
        model = Transformer(config)
        
        # Test complete pipeline
        text = "hello world"
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        
        import torch
        input_tensor = torch.tensor([token_ids])
        
        with torch.no_grad():
            outputs = model(input_tensor, input_tensor)
        
        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.shape[0], 1)  # batch size
        self.assertEqual(outputs.shape[2], config.vocab_size)  # vocab size


if __name__ == "__main__":
    unittest.main() 