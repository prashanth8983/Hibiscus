"""
Configuration management for the Transformer model.

This module provides configuration classes and utilities for managing
model hyperparameters, training settings, and data processing options.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
import yaml
import os
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the Transformer model architecture."""
    
    # Model architecture parameters
    vocab_size: int = 30000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    
    # Positional encoding
    pos_encoding_type: str = "sinusoidal"  # "sinusoidal" or "learned"
    
    # Layer normalization
    layer_norm_eps: float = 1e-6
    
    # Initialization
    init_std: float = 0.02
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")
        
        if self.pos_encoding_type not in ["sinusoidal", "learned"]:
            raise ValueError(f"pos_encoding_type must be 'sinusoidal' or 'learned', got {self.pos_encoding_type}")


@dataclass
class TrainingConfig:
    """Configuration for training the Transformer model."""
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    max_epochs: int = 100
    gradient_clip_val: float = 1.0
    
    # Optimization
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "cosine", "linear", "step"
    lr_decay_steps: Optional[int] = None
    lr_decay_rate: float = 0.1
    
    # Data processing
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    # Logging and checkpointing
    log_every_n_steps: int = 100
    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 1
    max_checkpoints: int = 5
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"
    
    # Distributed training
    use_ddp: bool = False
    ddp_backend: str = "nccl"
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        
        if self.gradient_clip_val <= 0:
            raise ValueError(f"gradient_clip_val must be positive, got {self.gradient_clip_val}")


@dataclass
class DataConfig:
    """Configuration for data processing and tokenization."""
    
    # Data paths
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    test_data_path: Optional[str] = None
    
    # Translation data paths (for machine translation tasks)
    src_data_path: Optional[str] = None
    tgt_data_path: Optional[str] = None
    
    # Tokenization
    tokenizer_type: str = "bpe"  # "bpe", "wordpiece", "char"
    vocab_size: int = 30000
    min_freq: int = 2
    
    # Text processing
    lowercase: bool = True
    remove_punctuation: bool = False
    max_length: int = 512
    truncation: bool = True
    padding: bool = True
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    # Data augmentation
    use_augmentation: bool = False
    augmentation_prob: float = 0.1
    
    def __post_init__(self):
        """Validate data configuration."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        
        if self.min_freq < 0:
            raise ValueError(f"min_freq must be non-negative, got {self.min_freq}")
        
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")


@dataclass
class Config:
    """Main configuration class that combines all sub-configurations."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment tracking
    experiment_name: str = "transformer_experiment"
    project_name: str = "hibiscus_transformer"
    
    # Logging
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
    use_tensorboard: bool = True
    
    # Random seed
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary."""
        # Extract sub-configurations
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        
        # Create main config
        main_config = config_dict.copy()
        main_config.pop("model", None)
        main_config.pop("training", None)
        main_config.pop("data", None)
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            **main_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "experiment_name": self.experiment_name,
            "project_name": self.project_name,
            "log_dir": self.log_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "use_wandb": self.use_wandb,
            "use_tensorboard": self.use_tensorboard,
            "seed": self.seed
        }
    
    def save(self, config_path: Union[str, Path]):
        """Save configuration to a YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def __post_init__(self):
        """Create necessary directories."""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


# Predefined configurations for common model sizes
def get_small_config() -> Config:
    """Get configuration for a small transformer model."""
    return Config(
        model=ModelConfig(
            vocab_size=30000,
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.1
        ),
        training=TrainingConfig(
            batch_size=32,
            learning_rate=1e-4,
            warmup_steps=4000,
            max_epochs=100
        )
    )


def get_medium_config() -> Config:
    """Get configuration for a medium transformer model."""
    return Config(
        model=ModelConfig(
            vocab_size=50000,
            d_model=768,
            n_heads=12,
            n_layers=12,
            d_ff=3072,
            max_seq_len=512,
            dropout=0.1
        ),
        training=TrainingConfig(
            batch_size=16,
            learning_rate=5e-5,
            warmup_steps=8000,
            max_epochs=200
        )
    )


def get_large_config() -> Config:
    """Get configuration for a large transformer model."""
    return Config(
        model=ModelConfig(
            vocab_size=50000,
            d_model=1024,
            n_heads=16,
            n_layers=24,
            d_ff=4096,
            max_seq_len=512,
            dropout=0.1
        ),
        training=TrainingConfig(
            batch_size=8,
            learning_rate=1e-5,
            warmup_steps=16000,
            max_epochs=300
        )
    ) 