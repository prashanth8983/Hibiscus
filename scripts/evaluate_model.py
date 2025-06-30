#!/usr/bin/env python3
"""
Model evaluation script for the Transformer model.

This script provides comprehensive evaluation capabilities including
performance benchmarking, attention visualization, and model analysis.
"""

import argparse
import sys
from pathlib import Path
import json
import time

# Add the parent directory to the path to import the transformer package
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer import Transformer, Config, Tokenizer, TextDataset
from transformer.analysis import ModelAnalyzer, PerformanceAnalyzer, create_model_report
from transformer.data import collate_fn


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
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return model, tokenizer, config


def evaluate_performance(model: Transformer, tokenizer: Tokenizer, 
                        test_data_path: str, config: Config):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained transformer model
        tokenizer: Tokenizer for text processing
        test_data_path: Path to test data
        config: Model configuration
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Create test dataset
    test_dataset = TextDataset(
        test_data_path,
        config.data,
        tokenizer=tokenizer,
        split="test"
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.data.pin_memory,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )
    
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    total_predictions = 0
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            
            # Forward pass
            outputs = model(src_ids, tgt_ids)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_ids.view(-1))
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = outputs.argmax(dim=-1)
            mask = (tgt_ids != tokenizer.pad_token_id)
            correct_predictions += ((predictions == tgt_ids) & mask).sum().item()
            total_predictions += mask.sum().item()
            total_tokens += mask.sum().item()
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions
    }


def generate_text_samples(model: Transformer, tokenizer: Tokenizer, 
                         prompts: list, max_length: int = 50,
                         temperature: float = 1.0, top_k: int = 10):
    """
    Generate text samples from the model.
    
    Args:
        model: Trained transformer model
        tokenizer: Tokenizer for text processing
        prompts: List of prompt texts
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        
    Returns:
        List of generated texts
    """
    device = next(model.parameters()).device
    generated_texts = []
    
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            # Encode prompt
            prompt_ids = tokenizer.encode(prompt)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
            
            # Generate
            generated_ids = model.generate(
                prompt_tensor,
                max_len=max_length,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode
            generated_text = tokenizer.decode(generated_ids[0].tolist())
            generated_texts.append({
                'prompt': prompt,
                'generated': generated_text,
                'length': len(generated_ids[0])
            })
    
    return generated_texts


def benchmark_model(model: Transformer, batch_sizes: list = [1, 4, 8, 16],
                   seq_lengths: list = [64, 128, 256, 512]):
    """
    Benchmark model performance.
    
    Args:
        model: Transformer model to benchmark
        batch_sizes: List of batch sizes to test
        seq_lengths: List of sequence lengths to test
        
    Returns:
        Dictionary containing benchmark results
    """
    perf_analyzer = PerformanceAnalyzer(model)
    
    # Benchmark inference
    inference_results = perf_analyzer.benchmark_inference(
        batch_sizes=batch_sizes,
        seq_lengths=seq_lengths,
        num_runs=5
    )
    
    # Memory analysis
    memory_results = perf_analyzer.memory_usage_analysis(
        batch_size=1,
        seq_len=512
    )
    
    return {
        'inference_benchmarks': inference_results,
        'memory_analysis': memory_results
    }


def analyze_attention_patterns(model: Transformer, tokenizer: Tokenizer,
                              sample_texts: list, save_dir: str = "attention_plots"):
    """
    Analyze and visualize attention patterns.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer for text processing
        sample_texts: List of sample texts to analyze
        save_dir: Directory to save attention visualizations
        
    Returns:
        Dictionary containing attention analysis results
    """
    device = next(model.parameters()).device
    analyzer = ModelAnalyzer(model)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    attention_results = {}
    
    for i, text in enumerate(sample_texts):
        # Tokenize
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        
        # Create input tensors
        src_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
        tgt_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
        
        # Get attention weights
        attention_weights = analyzer.get_attention_weights(src_tensor, tgt_tensor)
        
        # Visualize attention for each layer and head
        for layer_name, layer_weights in attention_weights.items():
            for head_idx in range(layer_weights.size(1)):  # Number of heads
                # Create visualization
                analyzer.visualize_attention(
                    attention_weights=layer_weights,
                    src_tokens=tokens,
                    tgt_tokens=tokens,
                    layer=int(layer_name.split('_')[1]),
                    head=head_idx,
                    save_path=str(save_path / f"attention_text_{i}_layer_{layer_name}_head_{head_idx}.png")
                )
        
        attention_results[f"text_{i}"] = {
            'text': text,
            'tokens': tokens,
            'attention_layers': list(attention_weights.keys())
        }
    
    return attention_results


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
        "--test-data", 
        type=str, 
        default=None,
        help="Path to test data for evaluation"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--generate-samples", 
        action="store_true",
        help="Generate text samples"
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true",
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--analyze-attention", 
        action="store_true",
        help="Analyze attention patterns"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=50,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, args.config)
    
    print(f"Model loaded successfully!")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Model parameters: {model.count_parameters():,}")
    
    results = {}
    
    # Create model report
    print("\nGenerating model report...")
    report = create_model_report(model, str(output_dir / "model_report.txt"))
    print(report)
    results['model_report'] = report
    
    # Evaluate performance if test data provided
    if args.test_data:
        print(f"\nEvaluating performance on test data: {args.test_data}")
        eval_results = evaluate_performance(model, tokenizer, args.test_data, config)
        results['evaluation'] = eval_results
        
        print("Evaluation Results:")
        print(f"  Loss: {eval_results['loss']:.4f}")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  Perplexity: {eval_results['perplexity']:.4f}")
        print(f"  Total tokens: {eval_results['total_tokens']:,}")
    
    # Generate text samples
    if args.generate_samples:
        print("\nGenerating text samples...")
        sample_prompts = [
            "The transformer model",
            "Machine learning is",
            "Natural language processing",
            "Deep learning has",
            "Artificial intelligence"
        ]
        
        generated_samples = generate_text_samples(
            model, tokenizer, sample_prompts,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        results['generated_samples'] = generated_samples
        
        print("Generated Samples:")
        for sample in generated_samples:
            print(f"  Prompt: '{sample['prompt']}'")
            print(f"  Generated: '{sample['generated']}'")
            print(f"  Length: {sample['length']} tokens")
            print()
    
    # Run benchmarks
    if args.benchmark:
        print("\nRunning performance benchmarks...")
        benchmark_results = benchmark_model(model)
        results['benchmarks'] = benchmark_results
        
        print("Benchmark Results:")
        for key, value in benchmark_results['inference_benchmarks'].items():
            print(f"  {key}: {value['mean_time_ms']:.2f}ms ± {value['std_time_ms']:.2f}ms")
            print(f"    Throughput: {value['throughput']:.2f} samples/sec")
        
        if 'memory_analysis' in benchmark_results:
            mem = benchmark_results['memory_analysis']
            if 'error' not in mem:
                print(f"  Memory usage: {mem['memory_used_mb']:.2f} MB")
                print(f"  Peak memory: {mem['peak_memory_mb']:.2f} MB")
    
    # Analyze attention patterns
    if args.analyze_attention:
        print("\nAnalyzing attention patterns...")
        sample_texts = [
            "The transformer model is powerful for natural language processing.",
            "Attention mechanisms allow models to focus on relevant information."
        ]
        
        attention_results = analyze_attention_patterns(
            model, tokenizer, sample_texts,
            save_dir=str(output_dir / "attention_plots")
        )
        results['attention_analysis'] = attention_results
        
        print("Attention analysis completed!")
        print(f"Visualizations saved to: {output_dir / 'attention_plots'}")
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {results_file}")
    print(f"Model report saved to: {output_dir / 'model_report.txt'}")


if __name__ == "__main__":
    main() 