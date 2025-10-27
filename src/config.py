# config.py
from dataclasses import dataclass

@dataclass
class GPTConfig:
    """
    Configuration for the GPT model, using dataclass to match the Andrej implementation.
    """
    block_size: int = 1024  # max sequence length (context window)
    vocab_size: int = 50257 # number of tokens 
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of attention heads
    n_embd: int = 768       # embedding dimension
    # These fields are used by the modules (Block, Attention) even if not explicitly
    # listed in the core provided config snippet, as they are necessary for module setup.
    dropout: float = 0.1    
    bias: bool = True       
