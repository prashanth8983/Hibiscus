#!/usr/bin/env python3
"""
API server for the Hibiscus Transformer model.

This script provides a REST API interface for querying the transformer model,
including text generation, model information, and health checks.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add the parent directory to the path to import the transformer package
sys.path.append(str(Path(__file__).parent.parent))

from transformer import Transformer, ModelConfig, Tokenizer
from transformer.config import Config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and tokenizer
model: Optional[Transformer] = None
tokenizer: Optional[Tokenizer] = None
config: Optional[Config] = None
device: torch.device = torch.device('cpu')


def load_model(checkpoint_path: str, config_path: str) -> None:
    """Load the transformer model and tokenizer."""
    global model, tokenizer, config, device
    
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    logger.info(f"Loading config from: {config_path}")
    
    # Load configuration
    config = Config.from_yaml(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model_config = ModelConfig(
        vocab_size=config.data.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout
    )
    
    model = Transformer(model_config)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Model checkpoint loaded successfully")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}, using initialized model")
    
    model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = Tokenizer(
        tokenizer_type=config.data.tokenizer_type,
        vocab_size=config.data.vocab_size
    )
    
    # Try to load tokenizer from checkpoint directory
    tokenizer_dir = Path(checkpoint_path).parent / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer.load(str(tokenizer_dir))
        logger.info("Tokenizer loaded from checkpoint directory")
    else:
        logger.warning("Tokenizer not found, using default tokenizer")
    
    logger.info("Model and tokenizer loaded successfully")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'config_loaded': config is not None
    })


@app.route('/info', methods=['GET'])
def model_info():
    """Get model information."""
    if model is None or config is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return jsonify({
        'model_name': 'Hibiscus Transformer',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_config': {
            'vocab_size': config.data.vocab_size,
            'd_model': config.model.d_model,
            'n_heads': config.model.n_heads,
            'n_layers': config.model.n_layers,
            'd_ff': config.model.d_ff,
            'max_seq_len': config.model.max_seq_len,
            'dropout': config.model.dropout
        },
        'device': str(device),
        'tokenizer_type': config.data.tokenizer_type
    })


@app.route('/generate', methods=['POST'])
def generate_text():
    """Generate text from a prompt."""
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model or tokenizer not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Generation parameters
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.8)
        top_k = data.get('top_k', 50)
        top_p = data.get('top_p', 0.9)
        pad_token_id = data.get('pad_token_id', 0)
        eos_token_id = data.get('eos_token_id', 1)
        
        # Validate parameters
        if max_length <= 0 or max_length > 1000:
            return jsonify({'error': 'max_length must be between 1 and 1000'}), 400
        if temperature <= 0 or temperature > 2.0:
            return jsonify({'error': 'temperature must be between 0 and 2.0'}), 400
        if top_k is not None and top_k <= 0:
            return jsonify({'error': 'top_k must be positive'}), 400
        if top_p is not None and (top_p <= 0 or top_p > 1.0):
            return jsonify({'error': 'top_p must be between 0 and 1.0'}), 400
        
        # Tokenize prompt
        prompt_ids = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(
                prompt_tensor,
                max_len=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        generated_texts = [generated_text]
        
        return jsonify({
            'prompt': prompt,
            'generated_texts': generated_texts,
            'parameters': {
                'max_length': max_length,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'pad_token_id': pad_token_id,
                'eos_token_id': eos_token_id
            }
        })
        
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500


@app.route('/encode', methods=['POST'])
def encode_text():
    """Encode text to token IDs."""
    if tokenizer is None:
        return jsonify({'error': 'Tokenizer not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        token_ids = tokenizer.encode(text)
        
        return jsonify({
            'text': text,
            'token_ids': token_ids,
            'token_count': len(token_ids)
        })
        
    except Exception as e:
        logger.error(f"Error during encoding: {str(e)}")
        return jsonify({'error': f'Encoding failed: {str(e)}'}), 500


@app.route('/decode', methods=['POST'])
def decode_text():
    """Decode token IDs to text."""
    if tokenizer is None:
        return jsonify({'error': 'Tokenizer not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        token_ids = data.get('token_ids', [])
        if not token_ids:
            return jsonify({'error': 'No token_ids provided'}), 400
        
        text = tokenizer.decode(token_ids)
        
        return jsonify({
            'token_ids': token_ids,
            'text': text
        })
        
    except Exception as e:
        logger.error(f"Error during decoding: {str(e)}")
        return jsonify({'error': f'Decoding failed: {str(e)}'}), 500


@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text properties."""
    if tokenizer is None:
        return jsonify({'error': 'Tokenizer not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        token_ids = tokenizer.encode(text)
        
        return jsonify({
            'text': text,
            'analysis': {
                'character_count': len(text),
                'word_count': len(text.split()),
                'token_count': len(token_ids),
                'token_ids': token_ids,
                'vocabulary_coverage': len(set(token_ids)) / len(token_ids) if token_ids else 0
            }
        })
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


def main():
    """Main function to run the API server."""
    parser = argparse.ArgumentParser(description='Hibiscus Transformer API Server')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind the server to (default: 5000)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Load model
    try:
        load_model(args.checkpoint, args.config)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        sys.exit(1)
    
    # Run the server
    logger.info(f"Starting API server on {args.host}:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main() 