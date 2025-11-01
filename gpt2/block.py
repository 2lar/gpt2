"""
Transformer Decoder Block Module

Implements the core building block of GPT-2: a Transformer decoder layer.
Each block consists of:
1. Multi-head causal self-attention
2. Feed-forward network (MLP)
3. Layer normalization and residual connections

The architecture follows the "pre-norm" formulation where LayerNorm is applied
BEFORE the sub-layers (attention and MLP), which improves training stability.
"""
import torch.nn as nn
from gpt2.config import GPTConfig
from gpt2.attention import CausalSelfAttention

class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network (MLP).

    A simple two-layer fully-connected network applied to each position
    independently and identically. This is the "FFN" from the original
    transformer paper.

    Architecture: Linear -> GELU -> Linear -> Dropout
    Hidden dimension is expanded by 4x (standard transformer practice).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # First layer: expand from n_embd to 4*n_embd
        # This expansion gives the model more capacity to learn complex patterns
        # The 4x multiplier is a standard choice from the original transformer paper
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)

        # GELU activation: Gaussian Error Linear Unit
        # A smooth activation function that works better than ReLU for transformers
        # approximate='tanh' uses the tanh approximation for GPT-2 compatibility
        self.gelu = nn.GELU(approximate='tanh')

        # Second layer: project back from 4*n_embd to n_embd
        # Returns to the residual stream dimension
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        # Flag for special initialization (scales down residual path contributions)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass: expand -> activate -> project -> regularize.

        Args:
            x: (B, T, n_embd) input tensor

        Returns:
            (B, T, n_embd) output tensor
        """
        x = self.c_fc(x)      # (B, T, n_embd) -> (B, T, 4*n_embd)
        x = self.gelu(x)      # Apply non-linearity
        x = self.c_proj(x)    # (B, T, 4*n_embd) -> (B, T, n_embd)
        x = self.dropout(x)   # Regularization
        return x

class Block(nn.Module):
    """
    Transformer Decoder Block (GPT-style).

    The fundamental building block of GPT-2. Each block performs:
    1. Self-attention to aggregate information from previous tokens
    2. Feed-forward computation to process the aggregated information

    Uses "pre-norm" architecture: LayerNorm BEFORE each sub-layer.
    This differs from the original transformer which used "post-norm".
    Pre-norm is more stable for deep networks and easier to train.

    The residual connections (x = x + ...) are crucial for gradient flow
    in deep networks, allowing gradients to bypass layers during backprop.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Layer normalization before attention
        # Normalizes activations to have mean=0, variance=1 across embedding dimension
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # Multi-head causal self-attention
        # Allows tokens to gather information from previous context
        self.attn = CausalSelfAttention(config)

        # Layer normalization before MLP
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # Position-wise feed-forward network
        # Processes each token's representation independently
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Forward pass through the transformer block.

        The structure is:
            x = x + attention(norm(x))    # Communication: gather info from context
            x = x + mlp(norm(x))           # Computation: process the information

        This is called "pre-norm residual" architecture.

        Args:
            x: (B, T, n_embd) input tensor

        Returns:
            (B, T, n_embd) output tensor with same shape
        """
        # Attention block with residual connection
        # "Communication" phase: tokens look at previous tokens and aggregate info
        x = x + self.attn(self.ln_1(x))

        # MLP block with residual connection
        # "Computation" phase: process the information independently for each token
        x = x + self.mlp(self.ln_2(x))

        return x
