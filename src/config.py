import torch.nn as nn
import math
from dataclasses import dataclass

# --- Configuration Class ---
@dataclass
class GPTConfig:
    """Stores all hyperparameters for the GPT-2 model."""
    block_size: int = 1024 # max sequence length (T)
    vocab_size: int = 50257 # number of tokens (V): 50,000 BPE + 256 bytes + 1 <|endoftext|>
    n_layer: int = 12 # number of layers (N)
    n_head: int = 12 # number of heads (H)
    n_embd: int = 768 # embedding dimension (D)

# --- Initialization Utilities ---
def apply_nanogpt_scale(module: nn.Module, n_layer: int) -> float:
    """
    Calculates the scaling factor for residual projection layers based on nanoGPT logic.
    Returns 1.0 if no scaling is needed.
    """
    scale_factor = 1.0
    if isinstance(module, nn.Linear):
        # Check if the module is marked for residual scaling (c_proj layers)
        if hasattr(module, 'NANOGPT_SCALE_INIT') and module.NANOGPT_SCALE_INIT:
            # Residual Scaling from nanoGPT: std * (2 * N)**-0.5
            # We return the scaling factor to be multiplied with the base std (0.02)
            scale_factor = (2 * n_layer) ** -0.5
    return scale_factor
