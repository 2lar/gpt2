import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# Import necessary components from config.py
from config import GPTConfig, apply_nanogpt_scale

class CausalSelfAttention(nn.Module):
    """Multi-Headed Causal Self-Attention block (using Flash Attention if available)."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # QKV Projection (in batch) - using the efficiency trick
        # Weights initialized with std=0.02 implicitly by nn.Linear
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # Output Projection - marked for residual scaling
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Mark c_proj for special initialization scaling
        self.c_proj.NANOGPT_SCALE_INIT = True 

        # Check if Flash Attention is available (PyTorch 2.0+)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Flash Attention >= 2.0 not available, using manual attention.")
            # Causal mask buffer (only needed if not using Flash Attention)
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimensionality (D)
        
        # 1. Calculate Q, K, V for all heads in batch and move head forward
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention: (B, T, D) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # 2. Perform Causal Self-Attention
        if self.flash:
             # Use PyTorch 2.0 Flash Attention (efficient)
             attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True) # NOTE: Dropout handled later
        else:
             # Manual implementation of attention
             att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # Calculate scores
             causal_mask = self.bias[:, :, :T, :T]                             # Apply mask
             att = att.masked_fill(causal_mask == 0, float('-inf'))
             att = F.softmax(att, dim=-1)                                     # Normalize to weights
             # NOTE: Dropout would normally be applied here (att = self.attn_dropout(att))
             attn_output = att @ v                                            # Weighted sum of values

        # 3. Re-assemble head outputs and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C) 
        attn_output = self.c_proj(attn_output) # Final linear projection
        # NOTE: Dropout (self.resid_dropout) is applied after the residual connection in the Block
        
        return attn_output

