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

    # =========================================================================
    # Modern Attention Parameters (for attention.py)
    # =========================================================================

    # Number of KV heads for Grouped-Query Attention (GQA)
    # If n_kv_head == n_head: Standard Multi-Head Attention (MHA)
    # If n_kv_head < n_head: Grouped-Query Attention (GQA)
    # If n_kv_head == 1: Multi-Query Attention (MQA)
    # Example: n_head=32, n_kv_head=8 → 4 Q-heads share each KV-head (60% memory savings)
    # Used by: LLaMA 2/3 (n_kv_head=8), Mistral (n_kv_head=8)
    n_kv_head: int = 12  # Default: same as n_head (standard MHA)

    # Maximum position for RoPE precomputation
    # Rotary embeddings can extrapolate beyond this, but precomputing is faster
    # Modern models often use 2048, 4096, or even 32768
    max_position_embeddings: int = 2048

    # Sliding window size for sliding window attention (optional)
    # Only relevant if using SlidingWindowAttention from attention_modern.py
    # Set to None or very large value to disable sliding window
    # Used by: Mistral (window_size=4096)
    window_size: int = 4096
