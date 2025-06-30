"""
Professional Production-Grade Transformer Implementation

A comprehensive, production-ready implementation of the Transformer architecture
with modern best practices, extensive testing, and professional tooling.
"""

from .model import Transformer
from .config import ModelConfig
from .trainer import Trainer
from .data import TextDataset
from .tokenizer import Tokenizer

__version__ = "1.0.0"
__author__ = "Hibiscus Team"

__all__ = [
    "Transformer",
    "ModelConfig", 
    "Trainer",
    "TextDataset",
    "Tokenizer"
] 