"""
GPT-2 Model Implementation

This module contains the complete GPT-2 model architecture, assembling all
components (embeddings, transformer blocks, language modeling head) into a
full autoregressive language model.

Key features:
- Token and positional embeddings
- Stack of transformer decoder blocks
- Weight tying between input/output embeddings
- Support for loading pretrained weights from HuggingFace
- Optimizer configuration with weight decay
"""
import inspect
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

# Import necessary components
from gpt2.config import GPTConfig
from gpt2.block import Block

class GPT(nn.Module):
    """
    GPT-2 Transformer Language Model.

    A decoder-only transformer that predicts the next token given previous tokens.
    Consists of:
    1. Token + position embeddings
    2. Stack of transformer blocks (attention + MLP)
    3. Final layer norm
    4. Language modeling head (projects to vocabulary)

    Uses weight tying: the embedding matrix is shared with the output projection.
    This reduces parameters and helps the model learn better token representations.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Core transformer components stored in a ModuleDict
        # This follows GPT-2's naming convention for easy weight loading
        self.transformer = nn.ModuleDict(dict(
            # Token embeddings: converts token IDs to vectors
            # Shape: (vocab_size, n_embd) - one vector per token in vocabulary
            wte = nn.Embedding(config.vocab_size, config.n_embd),

            # Position embeddings: learned position encodings
            # Shape: (block_size, n_embd) - one vector per position
            # Unlike sinusoidal encodings, these are learned during training
            wpe = nn.Embedding(config.block_size, config.n_embd),

            # Dropout applied to the sum of token + position embeddings
            drop = nn.Dropout(config.dropout),

            # Stack of n_layer transformer blocks
            # Each block contains attention + MLP with residual connections
            # 'h' stands for "hidden layers" (GPT-2 naming convention)
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            # Final layer normalization applied after all blocks
            # Stabilizes the pre-logit representations
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # Language modeling head: projects hidden states to vocabulary logits
        # No bias term (following GPT-2 design)
        # Output shape per token: (vocab_size,) - one score per possible next token
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share parameters between input embeddings and output projection
        # This is a common technique that:
        # 1. Reduces total parameters significantly
        # 2. Forces the model to learn a consistent token representation space
        # 3. Improves performance (empirically validated in many papers)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights using custom scheme
        self.apply(self._init_weights)

        # Report parameter count (useful for comparing model sizes)
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
        """
        Initialize weights following GPT-2 paper specifications.

        Uses careful initialization to ensure training stability in deep networks:
        - Linear layers: Normal(0, 0.02) - standard deviation of 0.02
        - Residual projections: Scaled by 1/sqrt(2*n_layer) to prevent activation
          blow-up as depth increases
        - Embeddings: Normal(0, 0.02)
        - Biases: zeros

        This initialization scheme is critical for training deep transformers.
        """
        if isinstance(module, nn.Linear):
            # Base standard deviation for weight initialization
            std = 0.02

            # Special handling for residual projection layers (c_proj in attention/MLP)
            # These layers have NANOGPT_SCALE_INIT flag set during construction
            # Scale down by 1/sqrt(2*n_layer) to account for residual accumulation
            # Without this, activations explode in very deep networks
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            # Initialize biases to zero (if present)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Embedding weights initialized with same std as linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass: convert token IDs to next-token predictions.

        The flow is:
        1. Embed tokens and add positional information
        2. Pass through transformer blocks (self-attention + MLP)
        3. Apply final layer norm
        4. Project to vocabulary to get logits
        5. Optionally compute loss if targets provided

        Args:
            idx: Input token indices, shape (B, T)
                 B = batch size, T = sequence length
            targets: Target token indices for training, shape (B, T)
                     If provided, computes and returns loss

        Returns:
            logits: Predicted scores for next token, shape (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.size()

        # Validate sequence length doesn't exceed model's context window
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}")

        # Step 1: Get embeddings
        # Token embeddings: map each token ID to its learned vector
        token_embeddings = self.transformer.wte(idx)  # (B, T, n_embd)

        # Position embeddings: add learned positional information
        # Create position indices [0, 1, 2, ..., T-1] on the same device as input
        position_indices = torch.arange(0, T, dtype=torch.long, device=idx.device)
        position_embeddings = self.transformer.wpe(position_indices)  # (T, n_embd)

        # Combine token and position information (broadcasts position across batch)
        # This tells the model both WHAT the tokens are and WHERE they are
        x = token_embeddings + position_embeddings  # (B, T, n_embd)
        x = self.transformer.drop(x)  # Apply dropout for regularization

        # Step 2: Pass through transformer blocks
        # Each block refines the representation using attention + MLP
        for block in self.transformer.h:
            x = block(x)  # (B, T, n_embd) -> (B, T, n_embd)

        # Step 3: Final layer normalization
        # Stabilizes the representations before the output projection
        x = self.transformer.ln_f(x)  # (B, T, n_embd)

        # Step 4: Project to vocabulary to get next-token logits
        # For each position, produces a score for each possible next token
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Step 5: Compute loss if training targets provided
        loss = None
        if targets is not None:
            # Flatten batch and sequence dimensions for cross-entropy
            # This treats each position as an independent classification problem
            # logits: (B, T, vocab_size) -> (B*T, vocab_size)
            # targets: (B, T) -> (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                              device_type: str, *, master_process: bool = True) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with weight decay only on appropriate parameters.

        Weight decay is a regularization technique, but it should NOT be applied to:
        - Biases (1D tensors)
        - LayerNorm parameters (1D tensors)
        These are low-dimensional and don't benefit from L2 regularization.

        Weight decay SHOULD be applied to:
        - Weight matrices in Linear layers (2D tensors)
        - Embedding tables (2D tensors)

        Args:
            weight_decay: L2 regularization strength (typically 0.1)
            learning_rate: Initial learning rate (typically 6e-4 for GPT-2)
            device_type: 'cuda' or 'cpu' - affects optimizer selection
            master_process: Whether this is the main process (for logging)

        Returns:
            Configured AdamW optimizer with separate parameter groups
        """
        # Get all trainable parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate into decay/no-decay groups based on dimensionality
        # 2D+ parameters: weight matrices and embeddings - apply decay
        # 1D parameters: biases and layer norm weights - no decay
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        # Create parameter groups with different weight decay settings
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        # Log parameter counts (helps verify setup is correct)
        if master_process:
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Use fused AdamW kernel if available (faster on CUDA)
        # Fused kernels combine multiple operations for better performance
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")

        # AdamW = Adam with decoupled weight decay (better than L2 regularization)
        # betas=(0.9, 0.95): momentum parameters (0.95 for second moment is GPT-2 specific)
        # eps=1e-8: small constant for numerical stability
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),  # GPT-2 uses 0.95 for beta2 (not standard 0.999)
            eps=1e-8,
            fused=use_fused,
        )
        return optimizer

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
