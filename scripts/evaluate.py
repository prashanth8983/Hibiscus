#!/usr/bin/env python3
"""
Evaluation script for the Transformer model.

This script demonstrates how to evaluate a trained transformer model
and generate text samples.
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path to import the transformer package
sys.path.append(str(Path(__file__).parent.parent))

import torch
from transformer import Transformer, Config, Tokenizer


def load_model_and_tokenizer(checkpoint_path: str, config_path: str):
    """
    Load a trained model and tokenizer from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    # Load configuration
    config = Config.from_yaml(config_path)
    
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
    
    # Create model
    model = Transformer(config.model)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer, config


def generate_text_samples(model: Transformer, tokenizer: Tokenizer, 
                         prompts: list, max_length: int = 100,
                         temperature: float = 1.0, top_k: int = 50,
                         top_p: float = 0.9):
    """
    Generate text samples from prompts.
    
    Args:
        model: Trained transformer model
        tokenizer: Tokenizer for text processing
        prompts: List of text prompts
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        
    Returns:
        List of generated texts
    """
    model.eval()
    device = next(model.parameters()).device
    
    generated_texts = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], device=device)
            
            # Generate
            generated_ids = model.generate(
                input_tensor,
                max_len=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode
            generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
            generated_texts.append(generated_text)
    
    return generated_texts


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer model")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--prompts", 
        type=str, 
        nargs='+',
        default=["The transformer model", "Machine learning is", "Deep learning has"],
        help="Text prompts for generation"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=100,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file for generated text"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, args.config)
    model.to(device)
    
    print(f"Model loaded successfully")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size()}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()
    
    # Generate text samples
    print("Generating text samples...")
    generated_texts = generate_text_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=args.prompts,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # Display results
    print("\nGenerated Text Samples:")
    print("=" * 50)
    
    for i, (prompt, generated) in enumerate(zip(args.prompts, generated_texts)):
        print(f"\nSample {i + 1}:")
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print("-" * 30)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("Generated Text Samples\n")
            f.write("=" * 30 + "\n\n")
            
            for i, (prompt, generated) in enumerate(zip(args.prompts, generated_texts)):
                f.write(f"Sample {i + 1}:\n")
                f.write(f"Prompt: '{prompt}'\n")
                f.write(f"Generated: '{generated}'\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"\nResults saved to {args.output}")
    
    # Model statistics
    print("\nModel Statistics:")
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size()}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Maximum sequence length: {config.model.max_seq_len}")
    print(f"Model dimensions: {config.model.d_model}")
    print(f"Number of layers: {config.model.n_layers}")
    print(f"Number of attention heads: {config.model.n_heads}")


if __name__ == "__main__":
    main() 