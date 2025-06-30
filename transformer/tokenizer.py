"""
Tokenization utilities for the Transformer model.

This module provides tokenization functionality including BPE, WordPiece,
and character-level tokenization, along with vocabulary management.
"""

import re
import json
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path


class BaseTokenizer:
    """Base class for all tokenizers."""
    
    def __init__(self, vocab_size: int = 30000, min_freq: int = 2,
                 pad_token: str = "<pad>", unk_token: str = "<unk>",
                 bos_token: str = "<s>", eos_token: str = "</s>"):
        """
        Initialize base tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for tokens
            pad_token: Padding token
            unk_token: Unknown token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Special tokens
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]
        
        # Vocabulary
        self.vocab = {}
        self.reverse_vocab = {}
        
        # Tokenizer state
        self.is_trained = False
    
    def train(self, texts: List[str]) -> None:
        """
        Train the tokenizer on a list of texts.
        
        Args:
            texts: List of training texts
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a single text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        raise NotImplementedError("Subclasses must implement tokenize method")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = [self.reverse_vocab.get(token_id, self.unk_token) for token_id in token_ids]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        
        return self._detokenize(tokens)
    
    def _detokenize(self, tokens: List[str]) -> str:
        """
        Convert tokens back to text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Detokenized text
        """
        return " ".join(tokens)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save tokenizer to file.
        
        Args:
            path: Path to save tokenizer
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer_data = {
            'vocab': self.vocab,
            'reverse_vocab': self.reverse_vocab,
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'special_tokens': self.special_tokens,
            'is_trained': self.is_trained,
            'tokenizer_type': self.__class__.__name__
        }
        
        with open(path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseTokenizer":
        """
        Load tokenizer from file.
        
        Args:
            path: Path to tokenizer file
            
        Returns:
            Loaded tokenizer
        """
        path = Path(path)
        
        with open(path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        # Create tokenizer instance
        tokenizer = cls(
            vocab_size=tokenizer_data['vocab_size'],
            min_freq=tokenizer_data['min_freq'],
            pad_token=tokenizer_data['pad_token'],
            unk_token=tokenizer_data['unk_token'],
            bos_token=tokenizer_data['bos_token'],
            eos_token=tokenizer_data['eos_token']
        )
        
        # Restore state
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.reverse_vocab = tokenizer_data['reverse_vocab']
        tokenizer.special_tokens = tokenizer_data['special_tokens']
        tokenizer.is_trained = tokenizer_data['is_trained']
        
        return tokenizer


class CharacterTokenizer(BaseTokenizer):
    """Character-level tokenizer."""
    
    def train(self, texts: List[str]) -> None:
        """
        Train character tokenizer.
        
        Args:
            texts: List of training texts
        """
        # Count character frequencies
        char_counts = Counter()
        for text in texts:
            char_counts.update(text)
        
        # Add special tokens with high frequency to ensure inclusion
        for token in self.special_tokens:
            char_counts[token] = 999999  # Use large integer instead of float
        
        # Select most frequent characters
        vocab_items = char_counts.most_common(self.vocab_size)
        
        # Build vocabulary
        for i, (char, _) in enumerate(vocab_items):
            self.vocab[char] = i
            self.reverse_vocab[i] = char
        
        self.is_trained = True
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into characters.
        
        Args:
            text: Input text
            
        Returns:
            List of characters
        """
        return list(text)


class WordTokenizer(BaseTokenizer):
    """Word-level tokenizer with basic preprocessing."""
    
    def __init__(self, vocab_size: int = 30000, min_freq: int = 2,
                 pad_token: str = "<pad>", unk_token: str = "<unk>",
                 bos_token: str = "<s>", eos_token: str = "</s>",
                 lowercase: bool = True, remove_punctuation: bool = False):
        """
        Initialize word tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for tokens
            pad_token: Padding token
            unk_token: Unknown token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            lowercase: Whether to lowercase text
            remove_punctuation: Whether to remove punctuation
        """
        super().__init__(vocab_size, min_freq, pad_token, unk_token, bos_token, eos_token)
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
    
    def train(self, texts: List[str]) -> None:
        """
        Train word tokenizer.
        
        Args:
            texts: List of training texts
        """
        # Preprocess and tokenize texts
        word_counts = Counter()
        for text in texts:
            words = self._preprocess_text(text)
            word_counts.update(words)
        
        # Add special tokens with high frequency to ensure inclusion
        for token in self.special_tokens:
            word_counts[token] = 999999  # Use large integer instead of float
        
        # Filter by minimum frequency
        word_counts = {word: count for word, count in word_counts.items() 
                      if count >= self.min_freq}
        
        # Select most frequent words
        vocab_items = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab_items = vocab_items[:self.vocab_size]
        
        # Build vocabulary
        for i, (word, _) in enumerate(vocab_items):
            self.vocab[word] = i
            self.reverse_vocab[i] = word
        
        self.is_trained = True
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        words = self._preprocess_text(text)
        return [word if word in self.vocab else self.unk_token for word in words]
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed words
        """
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        return text.split()
    
    def _detokenize(self, tokens: List[str]) -> str:
        """
        Convert tokens back to text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Detokenized text
        """
        return " ".join(tokens)


class BPETokenizer(BaseTokenizer):
    """Byte Pair Encoding (BPE) tokenizer."""
    
    def __init__(self, vocab_size: int = 30000, min_freq: int = 2,
                 pad_token: str = "<pad>", unk_token: str = "<unk>",
                 bos_token: str = "<s>", eos_token: str = "</s>"):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for tokens
            pad_token: Padding token
            unk_token: Unknown token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
        """
        super().__init__(vocab_size, min_freq, pad_token, unk_token, bos_token, eos_token)
        self.merges = {}
        self.word_freqs = Counter()
    
    def train(self, texts: List[str]) -> None:
        """
        Train BPE tokenizer.
        
        Args:
            texts: List of training texts
        """
        # Initialize vocabulary with characters
        char_vocab = set()
        for text in texts:
            char_vocab.update(text)
        
        # Add special tokens
        for token in self.special_tokens:
            char_vocab.add(token)
        
        # Initialize vocabulary
        self.vocab = {char: i for i, char in enumerate(char_vocab)}
        self.reverse_vocab = {i: char for char, i in self.vocab.items()}
        
        # Count word frequencies
        for text in texts:
            words = text.split()
            for word in words:
                self.word_freqs[word] += 1
        
        # Perform BPE merges
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            if len(self.word_freqs) == 0:
                break
            
            # Find most frequent pair
            pair_freqs = self._get_pair_frequencies()
            if not pair_freqs:
                break
            
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            
            # Add merge to vocabulary
            merged_token = "".join(best_pair)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
                self.reverse_vocab[len(self.vocab) - 1] = merged_token
            
            # Update word frequencies
            self._merge_pair(best_pair)
        
        self.is_trained = True
    
    def _get_pair_frequencies(self) -> Dict[Tuple[str, str], int]:
        """Get frequencies of adjacent character pairs."""
        pair_freqs = Counter()
        for word, freq in self.word_freqs.items():
            chars = list(word)
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def _merge_pair(self, pair: Tuple[str, str]) -> None:
        """Merge a pair of characters in all words."""
        new_word_freqs = Counter()
        merged_token = "".join(pair)
        
        for word, freq in self.word_freqs.items():
            new_word = word.replace("".join(pair), merged_token)
            new_word_freqs[new_word] += freq
        
        self.word_freqs = new_word_freqs
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using BPE.
        
        Args:
            text: Input text
            
        Returns:
            List of BPE tokens
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before tokenization")
        
        tokens = []
        for word in text.split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using BPE."""
        if word in self.vocab:
            return [word]
        
        # Greedy tokenization
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            while start < end:
                token = word[start:end]
                if token in self.vocab:
                    tokens.append(token)
                    start = end
                    break
                end -= 1
            else:
                # Unknown token
                tokens.append(self.unk_token)
                start += 1
        
        return tokens


class Tokenizer:
    """
    Main tokenizer class that supports different tokenization strategies.
    """
    
    def __init__(self, tokenizer_type: str = "bpe", **kwargs):
        """
        Initialize tokenizer.
        
        Args:
            tokenizer_type: Type of tokenizer ("bpe", "word", "char")
            **kwargs: Additional arguments for specific tokenizer
        """
        if tokenizer_type == "bpe":
            self.tokenizer = BPETokenizer(**kwargs)
        elif tokenizer_type == "word":
            self.tokenizer = WordTokenizer(**kwargs)
        elif tokenizer_type == "char":
            self.tokenizer = CharacterTokenizer(**kwargs)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    def train(self, texts: List[str]) -> None:
        """Train the tokenizer."""
        self.tokenizer.train(texts)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return self.tokenizer.tokenize(text)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer."""
        self.tokenizer.save(path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Tokenizer":
        """Load tokenizer."""
        tokenizer_data = pickle.load(open(path, 'rb'))
        tokenizer_type = tokenizer_data.get('tokenizer_type', 'bpe')
        
        if tokenizer_type == 'BPETokenizer':
            base_tokenizer = BPETokenizer.load(path)
        elif tokenizer_type == 'WordTokenizer':
            base_tokenizer = WordTokenizer.load(path)
        elif tokenizer_type == 'CharacterTokenizer':
            base_tokenizer = CharacterTokenizer.load(path)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
        instance = cls(tokenizer_type.lower().replace('tokenizer', ''))
        instance.tokenizer = base_tokenizer
        return instance
    
    @property
    def vocab(self) -> Dict[str, int]:
        """Get vocabulary."""
        return self.tokenizer.vocab
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer.vocab)
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.tokenizer.vocab[self.tokenizer.pad_token]
    
    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.tokenizer.vocab[self.tokenizer.unk_token]
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.tokenizer.vocab[self.tokenizer.bos_token]
    
    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID."""
        return self.tokenizer.vocab[self.tokenizer.eos_token] 