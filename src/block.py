import torch
import torch.nn as nn
import math

# Import necessary components
from config import GPTConfig, apply_nanogpt_scale
from attention import CausalSelfAttention # Import the attention module

# --- GELU Activation ---
# Using the approximate='tanh' version consistent with nanoGPT/GPT-2
class NewGELU(nn.Module):
    """Gaussian Error Linear Unit (using the tanh approximation)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implementation from OpenAI's CLIP model source code
        # Appeared in the GPT-2 paper: https://arxiv.org/abs/1606.08415
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# --- MLP ---
class MLP(nn.Module):
    """The Multi-Layer Perceptron (Feed-Forward Network) component."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Expansion layer (D -> 4D)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = NewGELU() # Use the specific GELU variant
        # Contraction layer (4D -> D) - marked for residual scaling
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # Mark c_proj for special initialization scaling
        self.c_proj.NANOGPT_SCALE_INIT = True 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        # NOTE: Dropout is applied after the residual connection in the Block
        return x

# --- The Full Transformer Block ---
class Block(nn.Module):
    """A single GPT Decoder Block with Pre-LayerNorm and Residual Connections."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Layer Normalization 1 (before Attention)
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5) # Added eps for numerical stability
        # Causal Self-Attention module
        self.attn = CausalSelfAttention(config)
        # Layer Normalization 2 (before MLP)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5) # Added eps
        # MLP module
        self.mlp = MLP(config)
        
        # NOTE: Dropout layers (attn_pdrop, resid_pdrop) from original config
        # are typically applied AFTER the residual connection, 
        # so they should belong in the main model's forward pass or within the Block's forward pass after addition.
        # For simplicity based on nanoGPT, they are often omitted or only applied at the very end.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm -> Attention -> Residual Connection
        x = x + self.attn(self.ln_1(x))
        # Pre-LayerNorm -> MLP -> Residual Connection
        x = x + self.mlp(self.ln_2(x))
        return x
