"""
Causal Self-Attention Module

Implements the multi-head self-attention mechanism with causal masking,
which is the core component that allows transformers to model long-range
dependencies in sequences.

Key concepts:
- Self-attention: Each token attends to all previous tokens (and itself)
- Causal masking: Future tokens are masked to preserve autoregressive property
- Multi-head: Multiple attention operations run in parallel for richer representations
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt.config import GPTConfig

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    This implements the attention mechanism from "Attention Is All You Need" with
    causal masking for autoregressive generation (GPT-style).

    The attention formula is: Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
    Including the dropuot, this becomes:
        Attention(Q, K, V) = Dropout(softmax(Q·K^T / sqrt(d_k))) · V

    Where each token can only attend to previous tokens (causal/autoregressive).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # Ensure embedding dimension is divisible by number of heads
        # So each head gets an equal slice of the embedding dimension
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Combined QKV projection: more efficient than 3 separate Linear layers
        # Projects from n_embd -> 3*n_embd (for Q, K, V simultaneously)
        # This is a standard optimization used in most transformer implementations
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection: projects concatenated multi-head output back to n_embd
        # This is the "W^O" matrix in the original transformer paper
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Special flag for scaled initialization (explained in GPT-2 paper)
        # This helps stabilize training in deep networks by scaling down residual contributions
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # Regularization
        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

        # Store dimensions for reshaping
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head  # Dimension per attention head
        
    def forward(self, x: torch.Tensor, kv_cache=None, use_cache=False):
        """
        Forward pass of multi-head causal self-attention with optional KV caching.

        Args:
            x: Input tensor of shape (B, T, C)
               B = batch size, T = sequence length, C = n_embd
            kv_cache: Optional tuple (past_k, past_v) of cached keys and values
                     Each has shape (B, n_head, past_T, head_size)
            use_cache: Whether to return updated KV cache for generation

        Returns:
            If use_cache=False: Output tensor of shape (B, T, C)
            If use_cache=True: Tuple of (output tensor, new_kv_cache)
        """
        B, T, C = x.size()  # Batch, Time (sequence length), Channels (n_embd)

        # Step 1: Project input to Query, Key, Value
        # Shape: (B, T, C) -> (B, T, 3C)
        qkv = self.c_attn(x)

        # Split into separate Q, K, V tensors along the embedding dimension
        # Each has shape (B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Step 2: Reshape for multi-head attention
        # Split embedding dimension into (n_head, head_size) and move heads to batch dim
        # (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        # This allows each head to operate independently in parallel
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, head_size)

        # Step 2b: KV Cache - concatenate with past keys/values if provided
        if kv_cache is not None:
            past_k, past_v = kv_cache  # Each: (B, n_head, past_T, head_size)
            k = torch.cat([past_k, k], dim=2)  # Concatenate along sequence dimension
            v = torch.cat([past_v, v], dim=2)

        # Store the updated cache (full K, V including past + new)
        new_kv_cache = (k, v) if use_cache else None

        # Step 3: Compute scaled dot-product attention with causal masking
        # Uses PyTorch's optimized Flash Attention implementation
        # is_causal=True means token i can only attend to tokens <= i (autoregressive)
        # This prevents the model from "cheating" by looking at future tokens
        attn_dropout = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            # Inside of here there is the calculation of Q·K^T / sqrt(d_k) + mask
            # the Q.matmul(K.transpose(-2, -1)) part computes the raw attention scores and is in shape of
            #   (B, n_head, T, T)
            # the dropout within this part is applied to the attention weights after softmax
            q, k, v,
            is_causal=True,  # Apply causal mask (lower triangular)
            dropout_p=attn_dropout
        )
        # Output shape: (B, n_head, T, head_size)

        # Step 4: Re-assemble all head outputs
        # (B, n_head, T, head_size) -> (B, T, n_head, head_size) -> (B, T, C)
        # Transpose moves sequence length back to position 1
        # Contiguous + view concatenates all heads back into single n_embd dimension
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Step 5: Final output projection and dropout
        # Projects the concatenated multi-head output through W^O
        y = self.resid_dropout(self.c_proj(y))

        if use_cache:
            return y, new_kv_cache
        return y
