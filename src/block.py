"""
block.py: Implements the Transformer Decoder Block.
Requires: config.py, attention.py
"""
import torch.nn as nn
from config import GPTConfig
from attention import CausalSelfAttention

class MLP(nn.Module):
    """
    The Feed-Forward Network (FFN) component of the Block.
    It typically expands the hidden dimension by a factor of 4.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Expansion layer (e.g., 768 -> 3072)
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh') # Use the tanh approximation for GPT-2 compatibility
        # self.gelu    = nn.GELU() # Standard activation in modern transformers
        # Projection layer (e.g., 3072 -> 768)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    A single GPT Transformer Decoder Block.
    It uses a pre-LayerNorm architecture (LN applied before the sub-layers).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Layer Normalization before Attention
        self.ln_1 = nn.LayerNorm(config.n_embd)
        
        # Causal Self Attention sub-layer
        self.attn = CausalSelfAttention(config)
        
        # Layer Normalization before MLP
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        # MLP sub-layer
        self.mlp = MLP(config)

    def forward(self, x):
        # First sub-layer: Attention (with pre-norm and residual connection)
        x = x + self.attn(self.ln_1(x))
        
        # Second sub-layer: MLP (with pre-norm and residual connection)
        x = x + self.mlp(self.ln_2(x))
        return x
