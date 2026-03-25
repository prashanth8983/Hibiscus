"""
Hibiscus Transformer — A from-scratch Transformer implementation in PyTorch.
"""

from .model import Transformer, TransformerEncoder, TransformerDecoder
from .config import ModelConfig, Config
from .trainer import Trainer
from .data import TextDataset, TranslationDataset
from .tokenizer import Tokenizer

__version__ = "1.0.0"
__author__ = "Hibiscus Team"

__all__ = [
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "ModelConfig",
    "Config",
    "Trainer",
    "TextDataset",
    "TranslationDataset",
    "Tokenizer",
]
