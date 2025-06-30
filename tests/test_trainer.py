"""
Unit tests for training components.

This module contains tests for the training infrastructure including
optimizers, schedulers, and training loops.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import the transformer package
sys.path.append(str(Path(__file__).parent.parent))

from transformer.trainer import WarmupCosineScheduler, Trainer
from transformer.config import Config, ModelConfig, TrainingConfig, DataConfig


class TestWarmupCosineScheduler(unittest.TestCase):
    """Test cases for WarmupCosineScheduler."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock optimizer
        self.optimizer = MagicMock()
        self.optimizer.param_groups = [{'lr': 1e-4}]
        
        self.warmup_steps = 10
        self.total_steps = 100
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, self.warmup_steps, self.total_steps
        )
    
    def test_warmup_phase(self):
        """Test learning rate during warmup phase."""
        # During warmup, LR should increase linearly
        for step in range(1, self.warmup_steps + 1):
            self.scheduler.step()
            expected_lr = 1e-4 * (step / self.warmup_steps)
            actual_lr = self.optimizer.param_groups[0]['lr']
            self.assertAlmostEqual(actual_lr, expected_lr, places=10)
    
    def test_decay_phase(self):
        """Test learning rate during decay phase."""
        # Skip warmup
        for _ in range(self.warmup_steps):
            self.scheduler.step()
        
        # During decay, LR should decrease following cosine schedule
        prev_lr = self.optimizer.param_groups[0]['lr']
        for _ in range(5):  # Test a few decay steps
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            # LR should be decreasing
            self.assertLess(current_lr, prev_lr)
            prev_lr = current_lr
    
    def test_get_last_lr(self):
        """Test getting current learning rate."""
        lr_list = self.scheduler.get_last_lr()
        self.assertEqual(len(lr_list), 1)
        self.assertEqual(lr_list[0], 1e-4)


class TestTrainer(unittest.TestCase):
    """Test cases for Trainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create minimal config
        self.config = Config(
            model=ModelConfig(
                vocab_size=100,
                d_model=64,
                n_heads=4,
                n_layers=2,
                d_ff=256,
                max_seq_len=32
            ),
            training=TrainingConfig(
                batch_size=4,
                learning_rate=1e-4,
                warmup_steps=10,
                max_epochs=5
            ),
            data=DataConfig(
                train_data_path="test_data",
                val_data_path="test_data"
            )
        )
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('transformer.trainer.DataLoader')
    def test_trainer_initialization(self, mock_dataloader, mock_cuda):
        """Test trainer initialization."""
        # Create a real model instead of a mock
        from transformer import Transformer, ModelConfig
        model = Transformer(ModelConfig(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=32
        ))
        mock_train_loader = MagicMock()
        mock_dataloader.return_value = mock_train_loader
        
        # Create trainer
        trainer = Trainer(model, self.config, mock_train_loader)
        
        # Check that trainer was initialized correctly
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)
        self.assertEqual(trainer.train_config, self.config.training)
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('transformer.trainer.DataLoader')
    def test_create_optimizer(self, mock_dataloader, mock_cuda):
        """Test optimizer creation."""
        # Create a real model instead of a mock
        from transformer import Transformer, ModelConfig
        model = Transformer(ModelConfig(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=32
        ))
        mock_train_loader = MagicMock()
        mock_dataloader.return_value = mock_train_loader
        
        trainer = Trainer(model, self.config, mock_train_loader)
        
        # Test different optimizer types
        self.config.training.optimizer = "adam"
        trainer._create_optimizer()
        self.assertIsNotNone(trainer.optimizer)
        
        self.config.training.optimizer = "adamw"
        trainer._create_optimizer()
        self.assertIsNotNone(trainer.optimizer)
        
        self.config.training.optimizer = "sgd"
        trainer._create_optimizer()
        self.assertIsNotNone(trainer.optimizer)
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('transformer.trainer.DataLoader')
    def test_invalid_optimizer(self, mock_dataloader, mock_cuda):
        """Test invalid optimizer raises error."""
        # Create a real model instead of a mock
        from transformer import Transformer, ModelConfig
        model = Transformer(ModelConfig(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=32
        ))
        mock_train_loader = MagicMock()
        mock_dataloader.return_value = mock_train_loader
        
        # Create trainer with invalid optimizer
        self.config.training.optimizer = "invalid_optimizer"
        
        with self.assertRaises(ValueError):
            trainer = Trainer(model, self.config, mock_train_loader)


class TestTrainingConfig(unittest.TestCase):
    """Test cases for TrainingConfig validation."""
    
    def test_valid_config(self):
        """Test creating a valid training configuration."""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=1e-4,
            warmup_steps=4000,
            max_epochs=100,
            gradient_clip_val=1.0
        )
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.warmup_steps, 4000)
        self.assertEqual(config.max_epochs, 100)
        self.assertEqual(config.gradient_clip_val, 1.0)
    
    def test_invalid_batch_size(self):
        """Test that batch_size must be positive."""
        with self.assertRaises(ValueError):
            TrainingConfig(batch_size=0)
    
    def test_invalid_learning_rate(self):
        """Test that learning_rate must be positive."""
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=0)
    
    def test_invalid_warmup_steps(self):
        """Test that warmup_steps must be non-negative."""
        with self.assertRaises(ValueError):
            TrainingConfig(warmup_steps=-1)
    
    def test_invalid_max_epochs(self):
        """Test that max_epochs must be positive."""
        with self.assertRaises(ValueError):
            TrainingConfig(max_epochs=0)
    
    def test_invalid_gradient_clip_val(self):
        """Test that gradient_clip_val must be positive."""
        with self.assertRaises(ValueError):
            TrainingConfig(gradient_clip_val=0)


if __name__ == '__main__':
    unittest.main() 