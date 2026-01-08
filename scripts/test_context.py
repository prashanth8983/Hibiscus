#!/usr/bin/env python3
"""
Test script for transformer context handling.

This script tests how well the transformer model handles different context lengths
and context-aware text generation.
"""

import sys
import json
import requests
from pathlib import Path

# Global variable for API base URL
API_BASE_URL = "http://127.0.0.1:5000"

# Add the parent directory to the path to import the transformer package
sys.path.append(str(Path(__file__).parent.parent))

from transformer import Transformer, ModelConfig, Tokenizer
from transformer.config import Config


def test_context_lengths():
    """Test model with different context lengths."""
    print("🧪 Testing Context Length Handling")
    print("=" * 50)
    
    # Test prompts of different lengths
    test_prompts = [
        "Hello",  # Short
        "The quick brown fox jumps over the lazy dog",  # Medium
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.",  # Long
        "The transformer model is a neural network architecture that revolutionized natural language processing. It was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. The key innovation of the transformer is its use of self-attention mechanisms, which allow the model to weigh the importance of different words in a sentence when processing each word. This attention mechanism enables the model to capture long-range dependencies in text more effectively than previous architectures like recurrent neural networks (RNNs) and long short-term memory (LSTM) networks.",  # Very long
    ]
    
    lengths = ["Short", "Medium", "Long", "Very Long"]
    
    for i, (prompt, length_type) in enumerate(zip(test_prompts, lengths)):
        print(f"\n📝 {length_type} Context Test ({len(prompt)} characters, {len(prompt.split())} words)")
        print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json={
                    "prompt": prompt,
                    "max_length": 30,
                    "temperature": 0.8,
                    "top_k": 50,
                    "top_p": 0.9
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result["generated_texts"][0]
                print(f"✅ Generated: {generated_text}")
                print(f"   Parameters: max_length={result['parameters']['max_length']}, temp={result['parameters']['temperature']}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")


def test_context_awareness():
    """Test context-aware generation with different scenarios."""
    print("\n🧠 Testing Context Awareness")
    print("=" * 50)
    
    context_scenarios = [
        {
            "name": "Technical Context",
            "prompt": "The transformer model uses attention mechanisms to",
            "expected": "technical terms related to transformers"
        },
        {
            "name": "Story Context", 
            "prompt": "Once upon a time, there was a brave knight who",
            "expected": "story continuation"
        },
        {
            "name": "Question Context",
            "prompt": "What is the capital of France?",
            "expected": "answer or related information"
        },
        {
            "name": "Code Context",
            "prompt": "def calculate_sum(a, b):",
            "expected": "code completion"
        }
    ]
    
    for scenario in context_scenarios:
        print(f"\n🎯 {scenario['name']}")
        print(f"Prompt: {scenario['prompt']}")
        print(f"Expected: {scenario['expected']}")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json={
                    "prompt": scenario["prompt"],
                    "max_length": 25,
                    "temperature": 0.7,
                    "top_k": 30,
                    "top_p": 0.8
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result["generated_texts"][0]
                print(f"✅ Generated: {generated_text}")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")


def test_parameter_variations():
    """Test generation with different parameter combinations."""
    print("\n⚙️ Testing Parameter Variations")
    print("=" * 50)
    
    base_prompt = "The future of artificial intelligence"
    
    parameter_sets = [
        {"temperature": 0.1, "top_k": 10, "name": "Conservative"},
        {"temperature": 0.5, "top_k": 30, "name": "Balanced"},
        {"temperature": 1.0, "top_k": 100, "name": "Creative"},
        {"temperature": 1.5, "top_k": 200, "name": "Very Creative"}
    ]
    
    for params in parameter_sets:
        print(f"\n🎲 {params['name']} Generation (temp={params['temperature']}, top_k={params['top_k']})")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json={
                    "prompt": base_prompt,
                    "max_length": 20,
                    "temperature": params["temperature"],
                    "top_k": params["top_k"],
                    "top_p": 0.9
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result["generated_texts"][0]
                print(f"✅ Generated: {generated_text}")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")


def test_tokenization_analysis():
    """Test tokenization and vocabulary coverage."""
    print("\n🔤 Testing Tokenization Analysis")
    print("=" * 50)
    
    test_texts = [
        "Hello world",
        "The transformer model is amazing",
        "Machine learning and artificial intelligence are transforming the world",
        "Python is a programming language used for data science and machine learning"
    ]
    
    for text in test_texts:
        print(f"\n📊 Analyzing: {text}")
        
        try:
            # Test encoding
            encode_response = requests.post(
                f"{API_BASE_URL}/encode",
                json={"text": text},
                timeout=10
            )
            
            if encode_response.status_code == 200:
                encode_result = encode_response.json()
                print(f"✅ Tokens: {encode_result['token_ids']}")
                print(f"   Token count: {encode_result['token_count']}")
                
                # Test analysis
                analyze_response = requests.post(
                    f"{API_BASE_URL}/analyze",
                    json={"text": text},
                    timeout=10
                )
                
                if analyze_response.status_code == 200:
                    analyze_result = analyze_response.json()
                    analysis = analyze_result["analysis"]
                    print(f"   Characters: {analysis['character_count']}")
                    print(f"   Words: {analysis['word_count']}")
                    print(f"   Vocabulary coverage: {analysis['vocabulary_coverage']:.2f}")
            else:
                print(f"❌ Encoding error: {encode_response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")


def test_model_limits():
    """Test model limits and edge cases."""
    print("\n🚧 Testing Model Limits")
    print("=" * 50)
    
    # Test very long prompt
    long_prompt = "This is a very long prompt that tests the model's ability to handle extended context. " * 10
    print(f"\n📏 Very Long Prompt Test ({len(long_prompt)} characters)")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={
                "prompt": long_prompt,
                "max_length": 10,
                "temperature": 0.8
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Successfully handled long prompt")
            print(f"   Generated: {result['generated_texts'][0]}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    
    # Test edge cases
    edge_cases = [
        {"prompt": "", "name": "Empty prompt"},
        {"prompt": "A", "name": "Single character"},
        {"prompt": "!@#$%^&*()", "name": "Special characters"},
        {"prompt": "1234567890", "name": "Numbers only"}
    ]
    
    for case in edge_cases:
        print(f"\n🔍 {case['name']}: '{case['prompt']}'")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json={
                    "prompt": case["prompt"],
                    "max_length": 15,
                    "temperature": 0.8
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Generated: {result['generated_texts'][0]}")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")


def main():
    """Run all context tests."""
    print("🧪 Hibiscus Transformer Context Testing Suite")
    print("=" * 60)
    
    # Check if API server is running (try both ports)
    api_urls = ["http://127.0.0.1:5001", "http://127.0.0.1:5000"]
    api_url = None
    
    for url in api_urls:
        try:
            health_response = requests.get(f"{url}/health", timeout=5)
            if health_response.status_code == 200:
                api_url = url
                break
        except requests.exceptions.RequestException:
            continue
    
    if api_url is None:
        print("❌ Cannot connect to API server. Please start it first:")
        print("python scripts/api.py --checkpoint checkpoints/improved_model.pt --config configs/improved.yaml --host 127.0.0.1 --port 5001")
        return
    
    print(f"✅ API server is running at {api_url}")
    
    # Update all requests to use the correct URL
    global API_BASE_URL
    API_BASE_URL = api_url
    
    # Run all tests
    test_context_lengths()
    test_context_awareness()
    test_parameter_variations()
    test_tokenization_analysis()
    test_model_limits()
    
    print("\n🎉 Context testing completed!")
    print("=" * 60)


if __name__ == '__main__':
    main() 