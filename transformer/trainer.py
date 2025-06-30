"""
Training utilities for the Transformer model.

This module provides training infrastructure including training loops,
optimization, logging, checkpointing, and evaluation utilities.
"""

import os
import time
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter

from .model import Transformer
from .config import Config, TrainingConfig
from .data import TextDataset, TranslationDataset
from .tokenizer import Tokenizer


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing.
    
    This scheduler implements the learning rate schedule used in the
    original Transformer paper and many modern transformer implementations.
    """
    
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int,
                 total_steps: int, min_lr: float = 0.0):
        """
        Initialize warmup cosine scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]


class Trainer:
    """
    Main trainer class for training transformer models.
    
    This class handles the complete training pipeline including
    training loops, validation, logging, and checkpointing.
    """
    
    def __init__(self, model: Transformer, config: Config,
                 train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 tokenizer: Optional[Tokenizer] = None):
        """
        Initialize trainer.
        
        Args:
            model: Transformer model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            tokenizer: Tokenizer for text generation (optional)
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        
        # Training configuration
        self.train_config = config.training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        total_steps = len(train_loader) * self.train_config.max_epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=self.train_config.warmup_steps,
            total_steps=total_steps,
            min_lr=0.0
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if self.train_config.use_amp else None
        
        # Initialize logging
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.train_config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                betas=(self.train_config.beta1, self.train_config.beta2),
                eps=self.train_config.eps,
                weight_decay=self.train_config.weight_decay
            )
        elif self.train_config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                betas=(self.train_config.beta1, self.train_config.beta2),
                eps=self.train_config.eps,
                weight_decay=self.train_config.weight_decay
            )
        elif self.train_config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.train_config.optimizer}")
    
    def _setup_logging(self):
        """Setup logging with TensorBoard and Weights & Biases."""
        # TensorBoard
        if self.config.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=self.config.log_dir)
        else:
            self.tb_writer = None
        
        # Weights & Biases
        if self.config.use_wandb:
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=self.config.to_dict()
            )
        else:
            wandb.init(mode="disabled")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.scaler is not None:
                with autocast():
                    if 'src_ids' in batch:
                        # Translation task
                        outputs = self.model(batch['src_ids'], batch['tgt_ids'][:, :-1])
                        targets = batch['tgt_ids'][:, 1:]
                    else:
                        # Language modeling task
                        outputs = self.model(batch['input_ids'], batch['target_ids'][:, :-1])
                        targets = batch['target_ids'][:, 1:]
                    
                    loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            else:
                if 'src_ids' in batch:
                    # Translation task
                    outputs = self.model(batch['src_ids'], batch['tgt_ids'][:, :-1])
                    targets = batch['tgt_ids'][:, 1:]
                else:
                    # Language modeling task
                    outputs = self.model(batch['input_ids'], batch['target_ids'][:, :-1])
                    targets = batch['target_ids'][:, 1:]
                
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            
            # Gradient clipping
            if self.train_config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.train_config.gradient_clip_val
                )
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Scheduler step
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_steps += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
            
            # Log metrics
            if self.global_step % self.train_config.log_every_n_steps == 0:
                self._log_metrics({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'train/epoch': self.current_epoch + batch_idx / len(self.train_loader)
                })
        
        return {
            'loss': total_loss / total_steps,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if 'src_ids' in batch:
                    # Translation task
                    outputs = self.model(batch['src_ids'], batch['tgt_ids'][:, :-1])
                    targets = batch['tgt_ids'][:, 1:]
                else:
                    # Language modeling task
                    outputs = self.model(batch['input_ids'], batch['target_ids'][:, :-1])
                    targets = batch['target_ids'][:, 1:]
                
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                
                total_loss += loss.item()
                total_steps += 1
        
        val_loss = total_loss / total_steps
        
        # Log validation metrics
        self._log_metrics({
            'val/loss': val_loss,
            'val/epoch': self.current_epoch
        })
        
        return {'loss': val_loss}
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to TensorBoard and Weights & Biases."""
        # TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, self.global_step)
        
        # Weights & Biases
        wandb.log(metrics, step=self.global_step)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.__dict__,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Remove old checkpoints if too many
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > self.train_config.max_checkpoints:
            for checkpoint_file in checkpoints[:-self.train_config.max_checkpoints]:
                checkpoint_file.unlink()
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state
        for key, value in checkpoint['scheduler_state_dict'].items():
            setattr(self.scheduler, key, value)
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, epochs: Optional[int] = None):
        """
        Train the model for the specified number of epochs.
        
        Args:
            epochs: Number of epochs to train (uses config if None)
        """
        if epochs is None:
            epochs = self.train_config.max_epochs
        
        print(f"Starting training for {epochs} epochs")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Log epoch metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics.get('loss', float('inf')),
                'learning_rate': train_metrics['learning_rate']
            }
            
            print(f"Epoch {epoch + 1}: Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Loss: {val_metrics.get('loss', float('inf')):.4f}")
            
            # Save checkpoint
            is_best = val_metrics.get('loss', float('inf')) < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if (epoch + 1) % self.train_config.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Close logging
        if self.tb_writer is not None:
            self.tb_writer.close()
        wandb.finish()
    
    def generate_text(self, prompt: str, max_length: int = 100,
                     temperature: float = 1.0, top_k: Optional[int] = None,
                     top_p: Optional[float] = None) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for text generation")
        
        self.model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_tensor,
                max_len=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        
        return generated_text 