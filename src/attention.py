"""
attention.py: Implements the Causal Self-Attention mechanism.
Requires: config.py
"""
import torch
import torch.nn as nn
from config import GPTConfig

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism, as used in a decoder-only
    (GPT-style) transformer block.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Combined Linear layer for Key, Query, and Value projections (QKV)
        # This is more efficient than three separate layers.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection (after attention calculation)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        
        # Causal mask (tril): A constant buffer that ensures each token only attends
        # to previous tokens in the sequence.
        self.register_buffer("bias", 
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimension (n_embd)

        # 1. Project to QKV
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 2. Reshape for Multi-Head Attention
        # (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 3. Compute Attention Scores (scaled dot-product)
        # att = (Q @ K^T) * (1/sqrt(head_size))
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))
        
        # 4. Apply Causal Masking
        # Mask future tokens by setting their attention scores to negative infinity
        # The mask is sliced to match the current sequence length T.
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # 5. Apply Softmax and Dropout
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 6. Apply Attention to Values
        # y = att @ V
        y = att @ v 
        
        # 7. Re-assemble heads
        # (B, n_head, T, head_size) -> (B, T, n_head, head_size) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Output projection and dropout (residual path)
        y = self.resid_dropout(self.c_proj(y))
        return y
