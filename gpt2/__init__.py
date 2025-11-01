"""
GPT-2 Implementation from Scratch

A educational reimplementation of GPT-2 following Andrej Karpathy's
"Build GPT from Scratch" series. This package contains a modular,
well-documented implementation suitable for learning transformer architectures.

Main components:
- GPTConfig: Model configuration dataclass
- CausalSelfAttention: Multi-head self-attention with causal masking
- Block: Transformer decoder block (attention + MLP)
- GPT: Complete GPT-2 model with training utilities
"""

from gpt2.config import GPTConfig
from gpt2.attention import CausalSelfAttention
from gpt2.block import Block, MLP
from gpt2.model import GPT

__all__ = [
    "GPTConfig",
    "CausalSelfAttention",
    "Block",
    "MLP",
    "GPT",
]

__version__ = "0.1.0"
