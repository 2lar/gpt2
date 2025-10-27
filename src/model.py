"""
model.py: The complete GPT (decoder-only) model.
Requires: config.py, block.py
"""
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# Import necessary components
from config import GPTConfig
from block import Block

class GPT(nn.Module):
    """
    The main GPT Transformer model class, assembling all components.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token embeddings (Wte)
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            
            # Position embeddings (Wpe)
            wpe = nn.Embedding(config.block_size, config.n_embd),
            
            # Initial dropout after embedding summation
            drop = nn.Dropout(config.dropout),
            
            # Stack of Transformer Blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            
            # Final layer norm (post-transformer stack, pre-head)
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Language Model Head: Projects the final hidden state to the vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share the token embeddings with the output layer weights
        self.transformer.wte.weight = self.lm_head.weight 

        # Initialize weights
        self.apply(self._init_weights)
        
        # Report the number of parameters
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f} Million")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Returns the total number of parameters, optionally excluding embedding layers."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract token and position embeddings (Wte and Wpe)
            n_params -= self.transformer.wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Standard Gaussian initialization for weights and zeros for biases/LayerNorm."""
        if isinstance(module, nn.Linear):
            # Normal distribution with stddev 0.02 is a common GPT initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Performs the forward pass of the model.
        
        Args:
            idx: Input token indices (B, T).
            targets: Target token indices (B, T) for loss calculation.
        """
        B, T = idx.size()
        
        # Check context window limit
        if T > self.config.block_size:
            print(f"Warning: Input sequence length ({T}) exceeds block_size ({self.config.block_size}).")
            # Truncate input to block_size if necessary
            idx = idx[:, :self.config.block_size]
            B, T = idx.size()

        # Get token and position embeddings
        token_embeddings = self.transformer.wte(idx) 
        
        # Get positional indices: (0, 1, 2, ..., T-1)
        position_indices = torch.arange(0, T, dtype=torch.long, device=idx.device)
        position_embeddings = self.transformer.wpe(position_indices) 
        
        # Sum embeddings and apply initial dropout
        x = self.transformer.drop(token_embeddings + position_embeddings)

        # Pass through the stack of blocks
        for block in self.transformer.h:
            x = block(x)

        # Final Layer Normalization
        x = self.transformer.ln_f(x)
        
        # Language Model Head to get logits
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Calculate cross-entropy loss by flattening the batch and sequence dimensions
            # logits: (B*T, vocab_size), targets: (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @staticmethod
    def normalize_hf_state_dict(state_dict: dict) -> dict:
        """Normalize a GPT-2 Hugging Face state_dict to expected key format."""
        if not any(k.startswith('transformer.') for k in state_dict.keys()):
            mapped = {}
            prefix_targets = ('wte.', 'wpe.', 'h.', 'ln_f')
            for key, value in state_dict.items():
                if key.startswith(prefix_targets):
                    mapped['transformer.' + key] = value
                else:
                    mapped[key] = value
            state_dict = mapped
        # Ensure lm_head.weight is present (HF checkpoints tie it to wte)
        if 'lm_head.weight' not in state_dict and 'transformer.wte.weight' in state_dict:
            state_dict['lm_head.weight'] = state_dict['transformer.wte.weight']
        # Drop attention bias buffers for compatibility with Hugging Face Modules
        for bias_key in [k for k in state_dict if k.endswith('.attn.bias')]:
            state_dict.pop(bias_key)
        return state_dict

    @classmethod
    def from_pretrained(cls, model_type: str, *, weights_path: str | None = None, local_files_only: bool = False):
        """
        Load Hugging Face GPT-2 weights into this implementation.
        Mirrors the approach from Karpathy's lecture: matches shapes/names and
        transposes Conv1D-style weights to Linear.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        if weights_path is None:
            from transformers import GPT2LMHeadModel

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        model = cls(GPTConfig(**config_args))
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        if weights_path is None:
            hf_model = GPT2LMHeadModel.from_pretrained(model_type, local_files_only=local_files_only)
            sd_hf = hf_model.state_dict()
        else:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            sd_hf = torch.load(weights_path, map_location='cpu')
        sd_hf = cls.normalize_hf_state_dict(sd_hf)
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not (k.endswith('.attn.masked_bias') or k.endswith('.attn.bias'))]

        # weights that need transpose (Conv1D -> Linear)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}")

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# --- Minimal runnable example ---
if __name__ == '__main__':
    # 1. Setup a small configuration for quick testing
    test_config = GPTConfig(
        block_size=64, 
        vocab_size=100, # Use a smaller vocab size for dummy data
        n_layer=2, 
        n_head=4, 
        n_embd=128,
        dropout=0.0
    )
    
    # 2. Instantiate the model
    print("--- Initializing GPT Model ---")
    model = GPT(test_config)
    model.eval()
    
    # 3. Create dummy input data (Batch size=4, Sequence length=32)
    B, T = 4, 32
    # Input indices: (B, T)
    input_indices = torch.randint(0, test_config.vocab_size, (B, T)) 
    # Target indices: (B, T) 
    target_indices = torch.randint(0, test_config.vocab_size, (B, T)) 
    
    print("-" * 50)
    print(f"Running test with inputs of shape: {input_indices.shape}")
    
    # 4. Perform the forward pass
    with torch.no_grad():
        logits, loss = model(input_indices, targets=target_indices)
    
    # 5. Print results
    print(f"Logits output shape: {logits.shape}")
    print(f"Calculated Loss: {loss.item():.4f}")
    print("Test successful!")
