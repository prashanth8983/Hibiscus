"""
Utility functions for the Transformer model.

This module provides various utility functions for model analysis,
visualization, and common operations used throughout the transformer implementation.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> str:
    """
    Get model size in a human-readable format.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size as a string (e.g., "1.2M", "500K")
    """
    total_params = count_parameters(model)
    
    if total_params >= 1e9:
        return f"{total_params / 1e9:.2f}B"
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.2f}M"
    elif total_params >= 1e3:
        return f"{total_params / 1e3:.2f}K"
    else:
        return str(total_params)


def create_attention_mask(seq_len: int, device: torch.device, 
                         causal: bool = True) -> torch.Tensor:
    """
    Create attention mask for causal or non-causal attention.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
        causal: Whether to create causal mask (for decoder)
        
    Returns:
        Attention mask tensor
    """
    if causal:
        # Create causal mask (lower triangular)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
    else:
        # Create full attention mask (no masking)
        mask = torch.zeros(seq_len, seq_len, device=device)
    
    return mask


def create_padding_mask(padding_mask: torch.Tensor, 
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Create padding mask for attention.
    
    Args:
        padding_mask: Boolean tensor indicating padded positions
        attention_mask: Optional existing attention mask
        
    Returns:
        Combined attention mask
    """
    # Create padding mask for attention
    seq_len = padding_mask.size(-1)
    pad_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    pad_mask = pad_mask.expand(-1, -1, seq_len, -1)    # (batch_size, 1, seq_len, seq_len)
    
    if attention_mask is not None:
        # Combine with existing attention mask
        return attention_mask + pad_mask
    else:
        return pad_mask


def positional_encoding(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """
    Create sinusoidal positional encoding.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        device: Device to create tensor on
        
    Returns:
        Positional encoding tensor
    """
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


def visualize_attention_weights(attention_weights: torch.Tensor, 
                               tokens: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> None:
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights tensor of shape (seq_len, seq_len)
        tokens: List of tokens for axis labels
        save_path: Path to save the visualization
    """
    # Convert to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(attention_weights, 
                xticklabels=tokens if tokens else 'auto',
                yticklabels=tokens if tokens else 'auto',
                cmap='Blues',
                annot=False,
                cbar=True)
    
    plt.title('Attention Weights')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_learning_curves(train_losses: List[float], 
                             val_losses: Optional[List[float]] = None,
                             save_path: Optional[str] = None) -> None:
    """
    Visualize learning curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity
    """
    return math.exp(loss)


def calculate_bleu_score(predictions: List[List[str]], 
                        references: List[List[List[str]]]) -> float:
    """
    Calculate BLEU score for machine translation evaluation.
    
    Args:
        predictions: List of predicted token sequences
        references: List of reference token sequences (can be multiple per prediction)
        
    Returns:
        BLEU score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
    except ImportError:
        print("NLTK is required for BLEU score calculation. Install with: pip install nltk")
        return 0.0
    
    smoothing = SmoothingFunction().method1
    
    total_bleu = 0.0
    for pred, refs in zip(predictions, references):
        # Convert tokens to strings if needed
        if isinstance(pred[0], str):
            pred_tokens = pred
        else:
            pred_tokens = [str(token) for token in pred]
        
        ref_tokens_list = []
        for ref in refs:
            if isinstance(ref[0], str):
                ref_tokens = ref
            else:
                ref_tokens = [str(token) for token in ref]
            ref_tokens_list.append(ref_tokens)
        
        bleu = sentence_bleu(ref_tokens_list, pred_tokens, smoothing_function=smoothing)
        total_bleu += bleu
    
    return total_bleu / len(predictions)


def gradient_clipping(model: nn.Module, max_norm: float = 1.0) -> None:
    """
    Apply gradient clipping to model parameters.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Freeze specific layers in the model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False


def unfreeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Unfreeze specific layers in the model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True


def get_layer_activations(model: nn.Module, 
                         input_tensor: torch.Tensor,
                         layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """
    Get activations from specific layers.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        layer_names: List of layer names to get activations from
        
    Returns:
        Dictionary of layer activations
    """
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if layer_names is None or name in layer_names:
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations


def analyze_model_complexity(model: nn.Module, 
                           input_shape: Tuple[int, ...]) -> Dict[str, Union[int, float]]:
    """
    Analyze model complexity including parameters and FLOPs.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor
        
    Returns:
        Dictionary with complexity metrics
    """
    # Count parameters
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate FLOPs (simplified)
    input_tensor = torch.randn(input_shape)
    
    def count_flops(module, input, output):
        if hasattr(module, 'weight'):
            if isinstance(module, nn.Linear):
                module.flops = input[0].numel() * module.weight.numel()
            elif isinstance(module, nn.Conv2d):
                module.flops = input[0].numel() * module.weight.numel()
    
    hooks = []
    for module in model.modules():
        hook = module.register_forward_hook(count_flops)
        hooks.append(hook)
    
    with torch.no_grad():
        model(input_tensor)
    
    total_flops = sum(getattr(module, 'flops', 0) for module in model.modules())
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'estimated_flops': total_flops,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }


def save_model_summary(model: nn.Module, 
                      save_path: str,
                      input_shape: Optional[Tuple[int, ...]] = None) -> None:
    """
    Save a detailed model summary to a file.
    
    Args:
        model: PyTorch model
        save_path: Path to save the summary
        input_shape: Shape of input tensor for complexity analysis
    """
    with open(save_path, 'w') as f:
        f.write("Model Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic info
        f.write(f"Model type: {type(model).__name__}\n")
        f.write(f"Total parameters: {count_parameters(model):,}\n")
        f.write(f"Model size: {get_model_size(model)}\n\n")
        
        # Layer-by-layer breakdown
        f.write("Layer Breakdown:\n")
        f.write("-" * 30 + "\n")
        
        total_params = 0
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                total_params += params
                f.write(f"{name}: {params:,} parameters\n")
        
        f.write(f"\nTotal parameters (verified): {total_params:,}\n")
        
        # Complexity analysis if input shape provided
        if input_shape:
            complexity = analyze_model_complexity(model, input_shape)
            f.write(f"\nComplexity Analysis:\n")
            f.write("-" * 20 + "\n")
            for key, value in complexity.items():
                f.write(f"{key}: {value}\n")
    
    print(f"Model summary saved to {save_path}") 