"""
Configuration module for GPT-2 model architecture.

This module defines the hyperparameters that control the model's structure,
following the original GPT-2 paper specifications.
"""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    GPT-2 Model Configuration.

    Defines the architecture hyperparameters for a GPT-2 style transformer.
    Default values correspond to GPT-2 "small" (124M parameters).

    Architecture sizes:
    - GPT-2 small:  n_layer=12, n_head=12, n_embd=768   (124M params)
    - GPT-2 medium: n_layer=24, n_head=16, n_embd=1024  (350M params)
    - GPT-2 large:  n_layer=36, n_head=20, n_embd=1280  (774M params)
    - GPT-2 xl:     n_layer=48, n_head=25, n_embd=1600  (1558M params)
    """

    # Context window - maximum number of tokens the model can attend to
    # GPT-2 uses 1024, though newer models use much larger contexts
    block_size: int = 1024

    # Vocabulary size - number of unique tokens in the tokenizer
    # 50257 = 50000 BPE merges + 256 byte tokens + 1 special <|endoftext|> token
    vocab_size: int = 50257

    # Depth - number of transformer decoder blocks stacked sequentially
    # More layers = more capacity to learn complex patterns
    n_layer: int = 12

    # Number of parallel attention heads in multi-head attention
    # Must divide evenly into n_embd (each head gets n_embd // n_head dimensions)
    n_head: int = 12

    # Embedding dimension - the "width" of the model
    # This is the size of the hidden state throughout the network
    n_embd: int = 768

    # Dropout probability for regularization (0.0 = no dropout)
    # Applied to attention weights and residual connections
    dropout: float = 0.0

    # Whether to use bias terms in Linear layers and LayerNorm
    # GPT-2 uses bias=True, but some modern models disable it
    bias: bool = True       
