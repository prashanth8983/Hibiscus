#!/usr/bin/env python3
"""
Basic usage example for the Transformer model.

This script demonstrates how to use the transformer implementation
for various tasks including model creation, training, and inference.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import the transformer package
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tempfile
import os

from transformer import (
    Transformer, 
    Config, 
    ModelConfig, 
    TrainingConfig, 
    DataConfig,
    Tokenizer,
    TextDataset
)
from transformer.trainer import Trainer
from transformer.data import create_data_loaders, collate_fn


def create_sample_data():
    """Create sample data for demonstration."""
    sample_texts = [
        "The transformer architecture has revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Multi-head attention enables models to attend to different aspects simultaneously.",
        "Positional encoding helps models understand the order of tokens in sequences.",
        "Self-attention allows each position to attend to all positions in the sequence.",
        "Cross-attention enables the decoder to attend to the encoder output.",
        "Layer normalization helps stabilize training of deep neural networks.",
        "Residual connections help with gradient flow in deep networks.",
        "Feed-forward networks provide non-linear transformations in transformers.",
        "The BERT model uses bidirectional transformers for language understanding.",
        "GPT models use unidirectional transformers for text generation.",
        "T5 is a unified text-to-text transformer model.",
        "RoBERTa is an improved version of BERT with better training.",
        "DistilBERT is a distilled version of BERT with fewer parameters.",
        "ALBERT reduces the number of parameters in BERT through parameter sharing."
    ]
    
    # Create temporary directory for data
    temp_dir = tempfile.mkdtemp()
    train_dir = os.path.join(temp_dir, "train")
    val_dir = os.path.join(temp_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Write training data
    with open(os.path.join(train_dir, "train.txt"), "w") as f:
        for i, text in enumerate(sample_texts * 10):  # Repeat texts
            f.write(f"{text} Sample {i}.\n")
    
    # Write validation data
    with open(os.path.join(val_dir, "val.txt"), "w") as f:
        for i, text in enumerate(sample_texts[:5] * 2):  # Fewer validation samples
            f.write(f"{text} Validation {i}.\n")
    
    return temp_dir


def demonstrate_model_creation():
    """Demonstrate creating and using a transformer model."""
    print("=== Model Creation Demo ===")
    
    # Create a small model configuration
    config = ModelConfig(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=64,
        dropout=0.1
    )
    
    # Create the model
    model = Transformer(config)
    print(f"Model created successfully!")
    print(f"Number of parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size()}")
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    src_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    tgt_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        outputs = model(src_ids, tgt_ids)
    
    print(f"Input shape: {src_ids.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Expected output shape: {(batch_size, seq_len, config.vocab_size)}")
    print("✓ Forward pass successful!\n")


def demonstrate_tokenizer():
    """Demonstrate tokenizer functionality."""
    print("=== Tokenizer Demo ===")
    
    # Create tokenizer
    tokenizer = Tokenizer(
        tokenizer_type="word",
        vocab_size=1000,
        min_freq=1,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>"
    )
    
    # Sample texts
    texts = [
        "The transformer is amazing.",
        "Attention mechanisms are powerful.",
        "Natural language processing with transformers."
    ]
    
    # Train tokenizer
    tokenizer.train(texts)
    print(f"Tokenizer trained with vocabulary size: {tokenizer.vocab_size}")
    
    # Test tokenization
    test_text = "The transformer is amazing."
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(token_ids)
    
    print(f"Original text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded text: {decoded}")
    print("✓ Tokenizer working correctly!\n")


def demonstrate_training():
    """Demonstrate training a transformer model."""
    print("=== Training Demo ===")
    
    # Create sample data
    data_dir = create_sample_data()
    print(f"Created sample data in: {data_dir}")
    
    # Create configuration
    config = Config(
        model=ModelConfig(
            vocab_size=500,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=32,
            dropout=0.1
        ),
        training=TrainingConfig(
            batch_size=4,
            learning_rate=1e-4,
            warmup_steps=10,
            max_epochs=2,  # Small number for demo
            gradient_clip_val=1.0,
            log_every_n_steps=5
        ),
        data=DataConfig(
            train_data_path=os.path.join(data_dir, "train"),
            val_data_path=os.path.join(data_dir, "val"),
            tokenizer_type="word",
            vocab_size=500,
            min_freq=1,
            max_length=32,
            batch_size=4
        ),
        experiment_name="demo_training",
        use_wandb=False,
        use_tensorboard=False
    )
    
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=0,  # Use 0 for demo
        pin_memory=config.data.pin_memory,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.data.pin_memory,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )
    
    # Create model
    model = Transformer(config.model)
    
    # Create trainer
    trainer = Trainer(model, config, train_loader, val_loader, tokenizer)
    
    # Train for a few epochs
    print("Starting training...")
    trainer.train(epochs=2)
    print("✓ Training completed!\n")
    
    # Clean up
    import shutil
    shutil.rmtree(data_dir)


def demonstrate_inference():
    """Demonstrate text generation with a trained model."""
    print("=== Inference Demo ===")
    
    # Create a small model for demonstration
    config = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=32,
        dropout=0.1
    )
    
    model = Transformer(config)
    model.eval()
    
    # Create a simple tokenizer for demo
    tokenizer = Tokenizer(
        tokenizer_type="char",
        vocab_size=100,
        min_freq=1
    )
    
    # Train on simple data
    sample_texts = ["hello", "world", "test", "demo"]
    tokenizer.train(sample_texts)
    
    # Generate text
    prompt = "hello"
    print(f"Generating text from prompt: '{prompt}'")
    
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_tensor,
            max_len=10,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"Generated text: '{generated_text}'")
    print("✓ Text generation successful!\n")


def demonstrate_configuration():
    """Demonstrate configuration management."""
    print("=== Configuration Demo ===")
    
    # Create configuration from scratch
    config = Config(
        model=ModelConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=8,
            n_layers=4,
            d_ff=1024,
            max_seq_len=128,
            dropout=0.1
        ),
        training=TrainingConfig(
            batch_size=16,
            learning_rate=1e-4,
            warmup_steps=1000,
            max_epochs=50
        ),
        data=DataConfig(
            train_data_path="data/train",
            val_data_path="data/val",
            tokenizer_type="bpe",
            vocab_size=1000
        ),
        experiment_name="demo_experiment"
    )
    
    print("Configuration created:")
    print(f"  Model: {config.model.d_model}d, {config.model.n_layers} layers, {config.model.n_heads} heads")
    print(f"  Training: {config.training.batch_size} batch size, {config.training.learning_rate} lr")
    print(f"  Data: {config.data.tokenizer_type} tokenizer, {config.data.vocab_size} vocab size")
    
    # Save configuration
    config_path = "demo_config.yaml"
    config.save(config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Load configuration
    loaded_config = Config.from_yaml(config_path)
    print(f"Configuration loaded from: {config_path}")
    
    # Clean up
    os.remove(config_path)
    print("✓ Configuration management working!\n")


def main():
    """Run all demonstrations."""
    print("🚀 Transformer Implementation Demo")
    print("=" * 50)
    
    try:
        demonstrate_model_creation()
        demonstrate_tokenizer()
        demonstrate_training()
        demonstrate_inference()
        demonstrate_configuration()
        
        print("🎉 All demonstrations completed successfully!")
        print("\nThis transformer implementation includes:")
        print("  ✓ Complete encoder-decoder architecture")
        print("  ✓ Multi-head attention mechanisms")
        print("  ✓ Positional encoding")
        print("  ✓ Professional training infrastructure")
        print("  ✓ Comprehensive tokenization")
        print("  ✓ Configuration management")
        print("  ✓ Text generation capabilities")
        print("  ✓ Extensive testing framework")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 