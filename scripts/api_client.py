#!/usr/bin/env python3
"""
Client script for the Hibiscus Transformer API.

This script demonstrates how to interact with the API endpoints
for text generation, encoding, decoding, and analysis.
"""

import json
import requests
import argparse
from typing import Dict, Any


class HibiscusAPIClient:
    """Client for the Hibiscus Transformer API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """Initialize the API client."""
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        response = requests.get(f"{self.base_url}/info")
        return response.json()
    
    def generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text from a prompt."""
        data = {
            'prompt': prompt,
            'max_length': kwargs.get('max_length', 100),
            'temperature': kwargs.get('temperature', 0.8),
            'top_k': kwargs.get('top_k', 50),
            'top_p': kwargs.get('top_p', 0.9),
            'pad_token_id': kwargs.get('pad_token_id', 0),
            'eos_token_id': kwargs.get('eos_token_id', 1)
        }
        
        response = requests.post(f"{self.base_url}/generate", json=data)
        return response.json()
    
    def encode_text(self, text: str) -> Dict[str, Any]:
        """Encode text to token IDs."""
        data = {'text': text}
        response = requests.post(f"{self.base_url}/encode", json=data)
        return response.json()
    
    def decode_text(self, token_ids: list) -> Dict[str, Any]:
        """Decode token IDs to text."""
        data = {'token_ids': token_ids}
        response = requests.post(f"{self.base_url}/decode", json=data)
        return response.json()
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text properties."""
        data = {'text': text}
        response = requests.post(f"{self.base_url}/analyze", json=data)
        return response.json()


def main():
    """Main function to demonstrate API usage."""
    parser = argparse.ArgumentParser(description='Hibiscus Transformer API Client')
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:5000',
        help='API base URL (default: http://localhost:5000)'
    )
    parser.add_argument(
        '--action',
        type=str,
        choices=['health', 'info', 'generate', 'encode', 'decode', 'analyze'],
        default='generate',
        help='Action to perform (default: generate)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='The transformer model',
        help='Text prompt for generation (default: The transformer model)'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Text for encoding/decoding/analysis'
    )
    parser.add_argument(
        '--tokens',
        type=str,
        help='Comma-separated token IDs for decoding'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum generation length (default: 100)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling parameter (default: 50)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p sampling parameter (default: 0.9)'
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = HibiscusAPIClient(args.url)
    
    try:
        if args.action == 'health':
            result = client.health_check()
            print("Health Check:")
            print(json.dumps(result, indent=2))
            
        elif args.action == 'info':
            result = client.get_model_info()
            print("Model Information:")
            print(json.dumps(result, indent=2))
            
        elif args.action == 'generate':
            result = client.generate_text(
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            print("Text Generation:")
            print(json.dumps(result, indent=2))
            
        elif args.action == 'encode':
            text = args.text or args.prompt
            result = client.encode_text(text)
            print("Text Encoding:")
            print(json.dumps(result, indent=2))
            
        elif args.action == 'decode':
            if not args.tokens:
                print("Error: --tokens argument required for decode action")
                return
            token_ids = [int(t.strip()) for t in args.tokens.split(',')]
            result = client.decode_text(token_ids)
            print("Text Decoding:")
            print(json.dumps(result, indent=2))
            
        elif args.action == 'analyze':
            text = args.text or args.prompt
            result = client.analyze_text(text)
            print("Text Analysis:")
            print(json.dumps(result, indent=2))
            
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {args.url}")
        print("Make sure the API server is running with:")
        print(f"python scripts/api.py --checkpoint <path> --config <path> --host 0.0.0.0 --port 5000")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    main() 