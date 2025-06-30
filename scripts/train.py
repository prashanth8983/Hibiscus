#!/usr/bin/env python3
"""
Training script for the Transformer model.

This script demonstrates how to train a transformer model using the
professional implementation provided in this package.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import the transformer package
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer import Transformer, Config, Trainer, TextDataset, Tokenizer
from transformer.data import create_data_loaders


def create_sample_data(data_dir: str, num_samples: int = 1000):
    """
    Create sample training data for demonstration.
    
    Args:
        data_dir: Directory to create data in
        num_samples: Number of sample texts to create
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Sample texts for demonstration
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Transformers have revolutionized natural language processing.",
        "Deep learning models require large amounts of data.",
        "Neural networks are inspired by biological neurons.",
        "Attention mechanisms allow models to focus on relevant information.",
        "The transformer architecture was introduced in 2017.",
        "Self-attention is a key component of transformer models.",
        "Positional encoding helps models understand sequence order.",
        "Multi-head attention allows models to attend to different aspects.",
        "Feed-forward networks provide non-linear transformations.",
        "Layer normalization helps with training stability.",
        "Residual connections help with gradient flow.",
        "Dropout is used for regularization during training.",
        "The BERT model uses bidirectional transformers.",
        "GPT models use unidirectional transformers for generation.",
        "T5 is a unified text-to-text transformer model.",
        "RoBERTa is an improved version of BERT.",
        "DistilBERT is a distilled version of BERT.",
        "ALBERT reduces the number of parameters in BERT."
    ]
    
    # Create training data
    train_path = data_path / "train"
    train_path.mkdir(exist_ok=True)
    
    with open(train_path / "train.txt", "w") as f:
        for i in range(num_samples):
            # Repeat and vary the sample texts
            text = sample_texts[i % len(sample_texts)]
            if i > len(sample_texts):
                text += f" This is sample number {i}."
            f.write(text + "\n")
    
    # Create validation data
    val_path = data_path / "val"
    val_path.mkdir(exist_ok=True)
    
    with open(val_path / "val.txt", "w") as f:
        for i in range(num_samples // 5):  # 20% for validation
            text = sample_texts[i % len(sample_texts)]
            text += f" Validation sample {i}."
            f.write(text + "\n")
    
    print(f"Created sample data in {data_dir}")
    print(f"Training samples: {num_samples}")
    print(f"Validation samples: {num_samples // 5}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/small.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--create-sample-data", 
        action="store_true",
        help="Create sample data for demonstration"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Number of epochs to train (overrides config)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create sample data if requested
    if args.create_sample_data:
        create_sample_data(args.data_dir)
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        print("Available configurations:")
        for config_file in Path("configs").glob("*.yaml"):
            print(f"  - {config_file}")
        return
    
    config = Config.from_yaml(config_path)
    print(f"Loaded configuration from {config_path}")
    
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Create tokenizer
    tokenizer = Tokenizer(
        tokenizer_type=config.data.tokenizer_type,
        vocab_size=config.data.vocab_size,
        min_freq=config.data.min_freq,
        pad_token=config.data.pad_token,
        unk_token=config.data.unk_token,
        bos_token=config.data.bos_token,
        eos_token=config.data.eos_token
    )
    
    # Create datasets
    train_dataset = TextDataset(
        config.data.train_data_path,
        config.data,
        tokenizer=tokenizer,
        split="train"
    )
    
    val_dataset = TextDataset(
        config.data.val_data_path,
        config.data,
        tokenizer=tokenizer,
        split="val"
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create data loaders
    from transformer.data import collate_fn
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )
    
    # Create model
    model = Transformer(config.model)
    model.to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size()}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        print(f"Resumed training from checkpoint: {args.checkpoint}")
    
    # Train the model
    epochs = args.epochs if args.epochs is not None else config.training.max_epochs
    trainer.train(epochs=epochs)
    
    # Generate some sample text
    print("\nGenerating sample text:")
    prompts = [
        "The transformer model",
        "Machine learning is",
        "Deep learning has",
        "Neural networks can",
        "Attention mechanisms"
    ]
    
    for prompt in prompts:
        generated = trainer.generate_text(
            prompt=prompt,
            max_length=50,
            temperature=0.8,
            top_k=50
        )
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print()


if __name__ == "__main__":
    main() 