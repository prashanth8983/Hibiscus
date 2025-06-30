"""
Data loading and preprocessing utilities for the Transformer model.

This module provides dataset classes, data loaders, and preprocessing
utilities for training transformer models on text data.
"""

import os
import json
import pickle
from typing import List, Dict, Optional, Tuple, Union, Iterator
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from .tokenizer import Tokenizer
from .config import DataConfig


class TextDataset(Dataset):
    """
    Dataset for text data with tokenization and preprocessing.
    
    This dataset handles loading text files, tokenization, and
    creating training pairs for sequence-to-sequence tasks.
    """
    
    def __init__(self, data_path: Union[str, Path], config: DataConfig,
                 tokenizer: Optional[Tokenizer] = None, split: str = "train"):
        """
        Initialize text dataset.
        
        Args:
            data_path: Path to data directory or file
            config: Data configuration
            tokenizer: Pre-trained tokenizer (optional)
            split: Dataset split ("train", "val", "test")
        """
        self.config = config
        self.split = split
        self.data_path = Path(data_path)
        
        # Initialize or load tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer(
                tokenizer_type=config.tokenizer_type,
                vocab_size=config.vocab_size,
                min_freq=config.min_freq,
                pad_token=config.pad_token,
                unk_token=config.unk_token,
                bos_token=config.bos_token,
                eos_token=config.eos_token
            )
        
        # Load and preprocess data
        self.texts = self._load_texts()
        
        # Train tokenizer if not provided
        if tokenizer is None and split == "train":
            self.tokenizer.train(self.texts)
        
        # Tokenize all texts
        self.tokenized_texts = self._tokenize_texts()
        
        # Create training pairs
        self.pairs = self._create_pairs()
    
    def _load_texts(self) -> List[str]:
        """Load texts from data path."""
        texts = []
        
        if self.data_path.is_file():
            # Single file
            with open(self.data_path, 'r', encoding='utf-8') as f:
                if self.data_path.suffix == '.json':
                    data = json.load(f)
                    if isinstance(data, list):
                        texts = data
                    else:
                        texts = [data]
                else:
                    texts = f.readlines()
        elif self.data_path.is_dir():
            # Directory of files
            for file_path in self.data_path.glob('*.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.extend(f.readlines())
        
        # Clean and filter texts
        texts = [text.strip() for text in texts if text.strip()]
        
        # Apply text preprocessing
        if self.config.lowercase:
            texts = [text.lower() for text in texts]
        
        if self.config.remove_punctuation:
            import re
            texts = [re.sub(r'[^\w\s]', '', text) for text in texts]
        
        return texts
    
    def _tokenize_texts(self) -> List[List[int]]:
        """Tokenize all texts."""
        tokenized = []
        for text in self.texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) <= self.config.max_length:
                tokenized.append(tokens)
        
        return tokenized
    
    def _create_pairs(self) -> List[Tuple[List[int], List[int]]]:
        """Create training pairs for sequence-to-sequence tasks."""
        pairs = []
        
        for tokens in self.tokenized_texts:
            if len(tokens) < 2:
                continue
            
            # For language modeling, create sliding window pairs
            for i in range(1, len(tokens)):
                input_seq = tokens[:i]
                target_seq = tokens[1:i+1]
                
                # Truncate if too long
                if len(input_seq) > self.config.max_length - 1:
                    input_seq = input_seq[-(self.config.max_length - 1):]
                    target_seq = target_seq[-(self.config.max_length - 1):]
                
                pairs.append((input_seq, target_seq))
        
        return pairs
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        input_seq, target_seq = self.pairs[idx]
        
        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'target_ids': torch.tensor(target_seq, dtype=torch.long),
            'input_length': len(input_seq),
            'target_length': len(target_seq)
        }


class TranslationDataset(Dataset):
    """
    Dataset for machine translation tasks.
    
    This dataset handles parallel text data for source-target language pairs.
    """
    
    def __init__(self, src_path: Union[str, Path], tgt_path: Union[str, Path],
                 config: DataConfig, src_tokenizer: Optional[Tokenizer] = None,
                 tgt_tokenizer: Optional[Tokenizer] = None, split: str = "train"):
        """
        Initialize translation dataset.
        
        Args:
            src_path: Path to source language data
            tgt_path: Path to target language data
            config: Data configuration
            src_tokenizer: Source language tokenizer
            tgt_tokenizer: Target language tokenizer
            split: Dataset split
        """
        self.config = config
        self.split = split
        self.src_path = Path(src_path)
        self.tgt_path = Path(tgt_path)
        
        # Initialize tokenizers
        if src_tokenizer is not None:
            self.src_tokenizer = src_tokenizer
        else:
            self.src_tokenizer = Tokenizer(
                tokenizer_type=config.tokenizer_type,
                vocab_size=config.vocab_size,
                min_freq=config.min_freq
            )
        
        if tgt_tokenizer is not None:
            self.tgt_tokenizer = tgt_tokenizer
        else:
            self.tgt_tokenizer = Tokenizer(
                tokenizer_type=config.tokenizer_type,
                vocab_size=config.vocab_size,
                min_freq=config.min_freq
            )
        
        # Load parallel data
        self.src_texts, self.tgt_texts = self._load_parallel_texts()
        
        # Train tokenizers if needed
        if src_tokenizer is None and split == "train":
            self.src_tokenizer.train(self.src_texts)
        if tgt_tokenizer is None and split == "train":
            self.tgt_tokenizer.train(self.tgt_texts)
        
        # Tokenize texts
        self.src_tokenized, self.tgt_tokenized = self._tokenize_parallel_texts()
        
        # Create pairs
        self.pairs = self._create_translation_pairs()
    
    def _load_parallel_texts(self) -> Tuple[List[str], List[str]]:
        """Load parallel source and target texts."""
        src_texts = []
        tgt_texts = []
        
        # Load source texts
        with open(self.src_path, 'r', encoding='utf-8') as f:
            src_texts = [line.strip() for line in f if line.strip()]
        
        # Load target texts
        with open(self.tgt_path, 'r', encoding='utf-8') as f:
            tgt_texts = [line.strip() for line in f if line.strip()]
        
        # Ensure same length
        min_len = min(len(src_texts), len(tgt_texts))
        src_texts = src_texts[:min_len]
        tgt_texts = tgt_texts[:min_len]
        
        return src_texts, tgt_texts
    
    def _tokenize_parallel_texts(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Tokenize parallel texts."""
        src_tokenized = []
        tgt_tokenized = []
        
        for src_text, tgt_text in zip(self.src_texts, self.tgt_texts):
            src_tokens = self.src_tokenizer.encode(src_text, add_special_tokens=True)
            tgt_tokens = self.tgt_tokenizer.encode(tgt_text, add_special_tokens=True)
            
            if len(src_tokens) <= self.config.max_length and len(tgt_tokens) <= self.config.max_length:
                src_tokenized.append(src_tokens)
                tgt_tokenized.append(tgt_tokens)
        
        return src_tokenized, tgt_tokenized
    
    def _create_translation_pairs(self) -> List[Tuple[List[int], List[int]]]:
        """Create translation pairs."""
        pairs = []
        
        for src_tokens, tgt_tokens in zip(self.src_tokenized, self.tgt_tokenized):
            pairs.append((src_tokens, tgt_tokens))
        
        return pairs
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single translation example."""
        src_tokens, tgt_tokens = self.pairs[idx]
        
        return {
            'src_ids': torch.tensor(src_tokens, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_tokens, dtype=torch.long),
            'src_length': len(src_tokens),
            'tgt_length': len(tgt_tokens)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], 
               pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching data.
    
    Args:
        batch: List of data samples
        pad_token_id: Padding token ID
        
    Returns:
        Batched data
    """
    # Separate different types of data
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    
    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=pad_token_id)
    
    # Create attention masks
    input_mask = (input_ids_padded != pad_token_id).long()
    target_mask = (target_ids_padded != pad_token_id).long()
    
    return {
        'input_ids': input_ids_padded,
        'target_ids': target_ids_padded,
        'input_mask': input_mask,
        'target_mask': target_mask,
        'input_lengths': torch.tensor([len(ids) for ids in input_ids]),
        'target_lengths': torch.tensor([len(ids) for ids in target_ids])
    }


def translation_collate_fn(batch: List[Dict[str, torch.Tensor]], 
                          pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function for translation data.
    
    Args:
        batch: List of translation samples
        pad_token_id: Padding token ID
        
    Returns:
        Batched translation data
    """
    # Separate source and target data
    src_ids = [item['src_ids'] for item in batch]
    tgt_ids = [item['tgt_ids'] for item in batch]
    
    # Pad sequences
    src_ids_padded = pad_sequence(src_ids, batch_first=True, padding_value=pad_token_id)
    tgt_ids_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=pad_token_id)
    
    # Create attention masks
    src_mask = (src_ids_padded != pad_token_id).long()
    tgt_mask = (tgt_ids_padded != pad_token_id).long()
    
    return {
        'src_ids': src_ids_padded,
        'tgt_ids': tgt_ids_padded,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_lengths': torch.tensor([len(ids) for ids in src_ids]),
        'tgt_lengths': torch.tensor([len(ids) for ids in tgt_ids])
    }


def create_data_loaders(config: DataConfig, tokenizer: Optional[Tokenizer] = None,
                       src_tokenizer: Optional[Tokenizer] = None,
                       tgt_tokenizer: Optional[Tokenizer] = None) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer for single-language tasks
        src_tokenizer: Source language tokenizer for translation
        tgt_tokenizer: Target language tokenizer for translation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Determine if this is a translation task
    is_translation = hasattr(config, 'src_data_path') and hasattr(config, 'tgt_data_path')
    
    if is_translation:
        # Translation task
        train_dataset = TranslationDataset(
            config.src_data_path + "/train",
            config.tgt_data_path + "/train",
            config, src_tokenizer, tgt_tokenizer, "train"
        )
        
        val_dataset = TranslationDataset(
            config.src_data_path + "/val",
            config.tgt_data_path + "/val",
            config, src_tokenizer, tgt_tokenizer, "val"
        )
        
        test_dataset = None
        if config.test_data_path:
            test_dataset = TranslationDataset(
                config.src_data_path + "/test",
                config.tgt_data_path + "/test",
                config, src_tokenizer, tgt_tokenizer, "test"
            )
        
        collate_fn_to_use = translation_collate_fn
        pad_token_id = src_tokenizer.pad_token_id if src_tokenizer else 0
        
    else:
        # Single language task
        train_dataset = TextDataset(
            config.train_data_path, config, tokenizer, "train"
        )
        
        val_dataset = TextDataset(
            config.val_data_path, config, tokenizer, "val"
        )
        
        test_dataset = None
        if config.test_data_path:
            test_dataset = TextDataset(
                config.test_data_path, config, tokenizer, "test"
            )
        
        collate_fn_to_use = collate_fn
        pad_token_id = tokenizer.pad_token_id if tokenizer else 0
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=lambda batch: collate_fn_to_use(batch, pad_token_id)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=lambda batch: collate_fn_to_use(batch, pad_token_id)
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=lambda batch: collate_fn_to_use(batch, pad_token_id)
        )
    
    return train_loader, val_loader, test_loader


class BucketSampler(Sampler):
    """
    Sampler that groups similar length sequences together for efficient batching.
    
    This reduces padding and improves training efficiency.
    """
    
    def __init__(self, dataset: Dataset, bucket_size: int = 10, shuffle: bool = True):
        """
        Initialize bucket sampler.
        
        Args:
            dataset: Dataset to sample from
            bucket_size: Number of sequences per bucket
            shuffle: Whether to shuffle buckets
        """
        self.dataset = dataset
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        
        # Group sequences by length
        self.buckets = self._create_buckets()
    
    def _create_buckets(self) -> List[List[int]]:
        """Create buckets of similar length sequences."""
        # Get sequence lengths
        lengths = []
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            if 'input_length' in item:
                lengths.append((i, item['input_length']))
            elif 'src_length' in item:
                lengths.append((i, item['src_length']))
            else:
                lengths.append((i, 0))
        
        # Sort by length
        lengths.sort(key=lambda x: x[1])
        
        # Create buckets
        buckets = []
        for i in range(0, len(lengths), self.bucket_size):
            bucket = [idx for idx, _ in lengths[i:i + self.bucket_size]]
            buckets.append(bucket)
        
        return buckets
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over indices."""
        if self.shuffle:
            # Shuffle buckets and sequences within buckets
            np.random.shuffle(self.buckets)
            for bucket in self.buckets:
                np.random.shuffle(bucket)
        
        for bucket in self.buckets:
            for idx in bucket:
                yield idx
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.dataset) 