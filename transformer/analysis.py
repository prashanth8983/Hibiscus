"""
Analysis and visualization utilities for the Transformer model.

This module provides tools for analyzing model performance, visualizing
attention weights, and understanding model behavior.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .model import Transformer
from .config import ModelConfig


class ModelAnalyzer:
    """
    Comprehensive model analysis utilities.
    
    This class provides methods for analyzing transformer models including
    parameter statistics, attention visualization, and performance metrics.
    """
    
    def __init__(self, model: Transformer):
        """
        Initialize model analyzer.
        
        Args:
            model: Transformer model to analyze
        """
        self.model = model
        self.config = model.config
    
    def analyze_parameters(self) -> Dict[str, Union[int, float, str]]:
        """
        Analyze model parameters and return statistics.
        
        Returns:
            Dictionary containing parameter statistics
        """
        total_params = 0
        trainable_params = 0
        param_groups = {}
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
            
            # Group parameters by type
            param_type = name.split('.')[0] if '.' in name else name
            if param_type not in param_groups:
                param_groups[param_type] = 0
            param_groups[param_type] += param_count
        
        # Calculate model size in MB
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': model_size_mb,
            'parameter_groups': param_groups,
            'vocab_size': self.config.vocab_size,
            'd_model': self.config.d_model,
            'n_layers': self.config.n_layers,
            'n_heads': self.config.n_heads,
            'd_ff': self.config.d_ff
        }
    
    def get_attention_weights(self, src: torch.Tensor, tgt: torch.Tensor,
                             layer_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from the model.
        
        Args:
            src: Source input tensor
            tgt: Target input tensor
            layer_idx: Specific layer to extract from (None for all layers)
            
        Returns:
            Dictionary containing attention weights for each layer
        """
        self.model.eval()
        attention_weights = {}
        
        with torch.no_grad():
            # Forward pass with return_attention=True
            outputs = self.model(src, tgt, return_attention=True)
            
            if isinstance(outputs, tuple):
                _, attention_dict = outputs
                
                if layer_idx is not None:
                    # Return specific layer
                    if f'layer_{layer_idx}' in attention_dict:
                        attention_weights[f'layer_{layer_idx}'] = attention_dict[f'layer_{layer_idx}']
                else:
                    # Return all layers
                    attention_weights = attention_dict
        
        return attention_weights
    
    def visualize_attention(self, attention_weights: torch.Tensor, 
                           src_tokens: Optional[List[str]] = None,
                           tgt_tokens: Optional[List[str]] = None,
                           layer: int = 0, head: int = 0,
                           save_path: Optional[str] = None) -> None:
        """
        Visualize attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weight tensor
            src_tokens: Source token labels
            tgt_tokens: Target token labels
            layer: Layer index for visualization
            head: Head index for visualization
            save_path: Path to save the visualization
        """
        # Extract specific layer and head
        if attention_weights.dim() == 4:  # (batch, n_heads, seq_len, seq_len)
            attn = attention_weights[0, head].cpu().numpy()
        elif attention_weights.dim() == 5:  # (batch, n_layers, n_heads, seq_len, seq_len)
            attn = attention_weights[0, layer, head].cpu().numpy()
        else:
            raise ValueError(f"Unexpected attention weights shape: {attention_weights.shape}")
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn, annot=False, cmap='Blues', cbar=True)
        
        # Set labels if provided
        if src_tokens is not None:
            plt.xticks(np.arange(len(src_tokens)) + 0.5, src_tokens, rotation=45, ha='right')
        if tgt_tokens is not None:
            plt.yticks(np.arange(len(tgt_tokens)) + 0.5, tgt_tokens, rotation=0)
        
        plt.title(f'Attention Weights - Layer {layer}, Head {head}')
        plt.xlabel('Source Tokens')
        plt.ylabel('Target Tokens')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to: {save_path}")
        
        plt.show()
    
    def analyze_gradients(self) -> Dict[str, float]:
        """
        Analyze gradient statistics for debugging training.
        
        Returns:
            Dictionary containing gradient statistics
        """
        grad_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_stats[f'{name}_norm'] = grad.norm().item()
                grad_stats[f'{name}_mean'] = grad.mean().item()
                grad_stats[f'{name}_std'] = grad.std().item()
                grad_stats[f'{name}_min'] = grad.min().item()
                grad_stats[f'{name}_max'] = grad.max().item()
        
        return grad_stats
    
    def compute_model_complexity(self) -> Dict[str, Union[int, float]]:
        """
        Compute model complexity metrics.
        
        Returns:
            Dictionary containing complexity metrics
        """
        # Count operations for a forward pass
        batch_size = 1
        seq_len = self.config.max_seq_len
        
        # Embedding operations
        embedding_ops = batch_size * seq_len * self.config.d_model
        
        # Attention operations per layer
        attention_ops_per_layer = (
            batch_size * seq_len * seq_len * self.config.d_model * 2 +  # QK computation
            batch_size * seq_len * seq_len * self.config.d_model +      # Attention * V
            batch_size * seq_len * self.config.d_model * self.config.d_model  # Output projection
        )
        
        # Feed-forward operations per layer
        ff_ops_per_layer = (
            batch_size * seq_len * self.config.d_model * self.config.d_ff +  # First layer
            batch_size * seq_len * self.config.d_ff * self.config.d_model    # Second layer
        )
        
        # Total operations
        total_attention_ops = attention_ops_per_layer * self.config.n_layers
        total_ff_ops = ff_ops_per_layer * self.config.n_layers
        
        # Output projection
        output_ops = batch_size * seq_len * self.config.d_model * self.config.vocab_size
        
        total_ops = embedding_ops + total_attention_ops + total_ff_ops + output_ops
        
        return {
            'total_operations': total_ops,
            'operations_per_token': total_ops / (batch_size * seq_len),
            'attention_operations': total_attention_ops,
            'feedforward_operations': total_ff_ops,
            'embedding_operations': embedding_ops,
            'output_operations': output_ops
        }


class PerformanceAnalyzer:
    """
    Performance analysis utilities for transformer models.
    """
    
    def __init__(self, model: Transformer):
        """
        Initialize performance analyzer.
        
        Args:
            model: Transformer model to analyze
        """
        self.model = model
        self.device = next(model.parameters()).device
    
    def benchmark_inference(self, batch_sizes: List[int] = [1, 4, 8, 16],
                           seq_lengths: List[int] = [64, 128, 256, 512],
                           num_runs: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Benchmark inference performance across different batch sizes and sequence lengths.
        
        Args:
            batch_sizes: List of batch sizes to test
            seq_lengths: List of sequence lengths to test
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary containing benchmark results
        """
        self.model.eval()
        results = {}
        
        with torch.no_grad():
            for batch_size in batch_sizes:
                for seq_len in seq_lengths:
                    key = f"batch_{batch_size}_seq_{seq_len}"
                    results[key] = {}
                    
                    # Create dummy inputs
                    src = torch.randint(0, self.model.config.vocab_size, 
                                      (batch_size, seq_len), device=self.device)
                    tgt = torch.randint(0, self.model.config.vocab_size, 
                                      (batch_size, seq_len), device=self.device)
                    
                    # Warmup
                    for _ in range(3):
                        _ = self.model(src, tgt)
                    
                    # Benchmark
                    times = []
                    for _ in range(num_runs):
                        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        
                        if torch.cuda.is_available():
                            start_time.record()
                            _ = self.model(src, tgt)
                            end_time.record()
                            torch.cuda.synchronize()
                            times.append(start_time.elapsed_time(end_time))
                        else:
                            import time
                            start = time.time()
                            _ = self.model(src, tgt)
                            end = time.time()
                            times.append((end - start) * 1000)  # Convert to ms
                    
                    # Calculate statistics
                    times = np.array(times)
                    results[key]['mean_time_ms'] = float(np.mean(times))
                    results[key]['std_time_ms'] = float(np.std(times))
                    results[key]['min_time_ms'] = float(np.min(times))
                    results[key]['max_time_ms'] = float(np.max(times))
                    results[key]['throughput'] = batch_size / (np.mean(times) / 1000)  # samples per second
        
        return results
    
    def memory_usage_analysis(self, batch_size: int = 1, seq_len: int = 512) -> Dict[str, float]:
        """
        Analyze memory usage of the model.
        
        Args:
            batch_size: Batch size for analysis
            seq_len: Sequence length for analysis
            
        Returns:
            Dictionary containing memory usage statistics
        """
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available for memory analysis'}
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Measure memory before
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Create inputs
        src = torch.randint(0, self.model.config.vocab_size, 
                          (batch_size, seq_len), device=self.device)
        tgt = torch.randint(0, self.model.config.vocab_size, 
                          (batch_size, seq_len), device=self.device)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(src, tgt)
        
        # Measure memory after
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_used_mb': memory_after - memory_before,
            'peak_memory_mb': peak_memory,
            'model_parameters_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
        }


def create_model_report(model: Transformer, save_path: Optional[str] = None) -> str:
    """
    Create a comprehensive model report.
    
    Args:
        model: Transformer model to analyze
        save_path: Path to save the report
        
    Returns:
        Report text
    """
    analyzer = ModelAnalyzer(model)
    perf_analyzer = PerformanceAnalyzer(model)
    
    # Get analyses
    param_stats = analyzer.analyze_parameters()
    complexity = analyzer.compute_model_complexity()
    
    # Create report
    report = []
    report.append("=" * 60)
    report.append("TRANSFORMER MODEL ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Model architecture
    report.append("MODEL ARCHITECTURE:")
    report.append("-" * 20)
    report.append(f"Vocabulary Size: {param_stats['vocab_size']:,}")
    report.append(f"Model Dimensions: {param_stats['d_model']}")
    report.append(f"Number of Layers: {param_stats['n_layers']}")
    report.append(f"Number of Heads: {param_stats['n_heads']}")
    report.append(f"Feed-forward Dimensions: {param_stats['d_ff']}")
    report.append(f"Maximum Sequence Length: {model.config.max_seq_len}")
    report.append("")
    
    # Parameter statistics
    report.append("PARAMETER STATISTICS:")
    report.append("-" * 20)
    report.append(f"Total Parameters: {param_stats['total_parameters']:,}")
    report.append(f"Trainable Parameters: {param_stats['trainable_parameters']:,}")
    report.append(f"Non-trainable Parameters: {param_stats['non_trainable_parameters']:,}")
    report.append(f"Model Size: {param_stats['model_size_mb']:.2f} MB")
    report.append("")
    
    # Parameter groups
    report.append("PARAMETER GROUPS:")
    report.append("-" * 20)
    for group, count in param_stats['parameter_groups'].items():
        report.append(f"{group}: {count:,} parameters")
    report.append("")
    
    # Complexity analysis
    report.append("COMPLEXITY ANALYSIS:")
    report.append("-" * 20)
    report.append(f"Total Operations: {complexity['total_operations']:,}")
    report.append(f"Operations per Token: {complexity['operations_per_token']:,.0f}")
    report.append(f"Attention Operations: {complexity['attention_operations']:,}")
    report.append(f"Feed-forward Operations: {complexity['feedforward_operations']:,}")
    report.append("")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Model report saved to: {save_path}")
    
    return report_text


def plot_training_curves(log_file: str, save_path: Optional[str] = None) -> None:
    """
    Plot training curves from log file.
    
    Args:
        log_file: Path to training log file
        save_path: Path to save the plot
    """
    # This would parse training logs and create plots
    # Implementation depends on the logging format
    pass 