import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect # For configure_optimizers
from typing import Optional, Tuple

# Import components from our other files
from config import GPTConfig, apply_nanogpt_scale
from block import Block # This imports Block, which internally uses MLP and CausalSelfAttention

class GPT(nn.Module):
    """The full GPT model architecture."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # --- Core Transformer Components ---
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # NOTE: Dropout layer after embeddings - added for clarity
            drop = nn.Dropout(0.1), # Using a fixed dropout rate here, can be configurable
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, eps=1e-5), # Final LayerNorm
        ))

        # --- Language Model Head ---
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # --- Weight Tying ---
        self.transformer.wte.weight = self.lm_head.weight

        # --- Initialize Parameters ---
        self.apply(self._init_weights)
        print(f"GPT model initialized with {sum(p.numel() for p in self.parameters()):,} parameters.")


    def _init_weights(self, module: nn.Module):
        """Initializes weights using Gaussian(0, 0.02) and applies Residual Scaling."""
        if isinstance(module, nn.Linear):
            std = 0.02
            scale_factor = apply_nanogpt_scale(module, self.config.n_layer)
            if scale_factor != 1.0:
                std *= scale_factor
                
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
             if module.bias is not None: torch.nn.init.zeros_(module.bias)
             if module.weight is not None: torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device) # (T)
        
        # 1. Embeddings + Dropout
        tok_emb = self.transformer.wte(idx) # (B, T, D)
        pos_emb = self.transformer.wpe(pos) # (T, D)
        x = self.transformer.drop(tok_emb + pos_emb) # Apply dropout after summing embeddings
        
        # 2. Transformer Blocks
        for block in self.transformer.h:
            x = block(x) 
            
        # 3. Final LayerNorm
        x = self.transformer.ln_f(x) 
        
        # 4. Language Model Head
        logits = self.lm_head(x) # (B, T, V)
        
        # 5. Loss Calculation
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float, device_type: str) -> torch.optim.AdamW:
        """Configures AdamW optimizer with weight decay separation."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Optimizer: num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Optimizer: num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"Optimizer: using fused AdamW: {use_fused}")
            
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type: str) -> 'GPT':
        """Loads pretrained GPT-2 weights from Hugging Face."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained gpt: {model_type}")

        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 
        config_args['block_size'] = 1024 
        
        config = GPTConfig(**config_args)
        model = cls(config) # Use 'cls' for class method instantiation
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')] # Exclude mask buffer

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(('.attn.masked_bias', '.attn.bias'))]
        
        # Handle potential key mismatches and Conv1D transposition
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} HF != {len(sd_keys)} local"
        
        for k in sd_keys_hf:
            with torch.no_grad():
                if any(k.endswith(w) for w in transposed):
                    # Transpose Conv1D weights from HF format
                    assert sd_hf[k].shape[::-1] == sd[k].shape
                    sd[k].copy_(sd_hf[k].t())
                else:
                    # Direct copy for other parameters
                    assert sd_hf[k].shape == sd[k].shape
                    sd[k].copy_(sd_hf[k])
        print(f"Weights loaded successfully for {model_type}")
        return model

# --- Example Usage ---
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    config = GPTConfig() # Default GPT-2 small
    model = GPT(config)
    model.to(device)
    
    # Dummy forward pass
    B, T_in = 2, 8
    dummy_input = torch.randint(0, config.vocab_size, (B, T_in)).to(device)
    logits, loss = model(dummy_input)
    print(f"Logits shape: {logits.shape}") # Should be (B, T_in, V)
    print("Model forward pass successful.")
