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
import yaml
from datasets import load_dataset

# Add the parent directory to the path to import the transformer package
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer import Transformer, Trainer, TextDataset, Tokenizer
from transformer.config import Config
from transformer.data import create_data_loaders, HuggingFaceTextDataset


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
        help="Directory containing training data (for local files)"
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
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset from the Hugging Face Hub (e.g., 'wikitext')"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Optional dataset configuration for Hugging Face datasets (e.g., 'wikitext-103-raw-v1')"
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

    hf_dataset = None
    if args.dataset_name:
        print(f"Loading Hugging Face dataset: {args.dataset_name}")
        if args.dataset_config:
            hf_dataset = load_dataset(args.dataset_name, args.dataset_config)
        else:
            hf_dataset = load_dataset(args.dataset_name)
        
        # Train tokenizer on the Hugging Face dataset
        def text_iterator():
            for example in hf_dataset["train"]:
                yield example["text"]
        tokenizer.train_from_iterator(text_iterator())
        
        # Save the tokenizer
        os.makedirs("checkpoints", exist_ok=True)
        tokenizer_path = f"checkpoints/{Path(args.config).stem}_tokenizer.pkl"
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    else:
        # Train tokenizer on local data if no HF dataset is specified
        train_texts = []
        train_data_path = Path(config.data.train_data_path)
        if train_data_path.is_file():
            with open(train_data_path, 'r', encoding='utf-8') as f:
                train_texts.extend(f.readlines())
        elif train_data_path.is_dir():
            for file_path in train_data_path.glob('*.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    train_texts.extend(f.readlines())
        train_texts = [text.strip() for text in train_texts if text.strip()]
        if train_texts:
            tokenizer.train(train_texts)
        else:
            print("Warning: No local training data found for tokenizer training.")
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(config.data, tokenizer, hf_dataset=hf_dataset)
    
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
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