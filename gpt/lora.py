"""
LoRA (Low-Rank Adaptation) Implementation

LoRA adds trainable low-rank matrices to frozen pretrained weights.
Instead of fine-tuning all weights W, we keep W frozen and add a low-rank update:
    W' = W + BA
where B is (d × r) and A is (r × d), with r << d

This dramatically reduces:
- Trainable parameters (only A and B, not W)
- Memory usage (only need to store small A, B matrices)
- VRAM requirements (base model stays frozen)

For GPT-2 attention, we typically apply LoRA to Q and V projections.
"""
import torch
import torch.nn as nn
from typing import Optional


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps an existing Linear layer.

    Adds trainable low-rank matrices A and B to the frozen weight W.
    Forward: y = W·x + (B·A)·x = W·x + B·(A·x)

    The scaling factor alpha/r controls the magnitude of the LoRA contribution.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # Low-rank matrices A and B
        # A: (rank, in_features) - initialized with random Gaussian
        # B: (out_features, rank) - initialized with zeros (no effect initially)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Scaling factor: controls the magnitude of LoRA update
        # Common practice: scale = alpha / rank
        self.scaling = alpha / rank

        # Optional dropout on the LoRA path
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, original_weight: torch.Tensor, original_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass combining original frozen weights with LoRA adaptation.

        Args:
            x: Input tensor (B, T, in_features)
            original_weight: Frozen weight matrix (out_features, in_features)
            original_bias: Optional frozen bias (out_features,)

        Returns:
            Output tensor (B, T, out_features)
        """
        # Original frozen linear transformation
        output = torch.nn.functional.linear(x, original_weight, original_bias)

        # LoRA adaptation: x·A^T·B^T scaled by alpha/r
        # This is computed as: (x @ A.T) @ B.T
        lora_out = self.dropout(x @ self.lora_A.T) @ self.lora_B.T

        # Combine: y = Wx + scale * BAx
        output = output + lora_out * self.scaling

        return output


class LinearWithLoRA(nn.Module):
    """
    Linear layer with optional LoRA adaptation.

    This wraps a standard Linear layer and adds LoRA capability.
    The original weights are frozen, only LoRA parameters train.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        enable_lora: bool = True
    ):
        super().__init__()
        # Store the original linear layer (will be frozen)
        self.linear = linear
        self.enable_lora = enable_lora

        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # Add LoRA layer if enabled
        if enable_lora:
            self.lora = LoRALayer(
                in_features=linear.in_features,
                out_features=linear.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
        else:
            self.lora = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear layer with optional LoRA."""
        if self.enable_lora and self.lora is not None:
            return self.lora(x, self.linear.weight, self.linear.bias)
        else:
            return self.linear(x)

    def merge_weights(self):
        """
        Merge LoRA weights into the base weights for inference.
        After merging: W' = W + alpha/r * B·A
        This eliminates the need for separate LoRA computation during inference.
        """
        if self.lora is not None:
            # Compute the low-rank update: B @ A
            lora_weight = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling

            # Add to the frozen weight
            self.linear.weight.data += lora_weight

            # Disable LoRA now that weights are merged
            self.enable_lora = False


def apply_lora_to_model(model: nn.Module, rank: int = 8, alpha: float = 16.0, target_modules: list = None):
    """
    Apply LoRA to specific modules in a GPT model.

    By default, targets the Q and V projections in attention layers.
    This is a common choice that balances effectiveness and parameter efficiency.

    Args:
        model: The GPT model to modify
        rank: LoRA rank (r)
        alpha: LoRA scaling factor
        target_modules: List of module name patterns to target (e.g., ['c_attn'])

    Returns:
        Number of LoRA parameters added
    """
    if target_modules is None:
        # Default: target attention projections
        # c_attn contains Q, K, V - we'll replace the whole thing
        target_modules = ['c_attn']

    total_lora_params = 0

    # Iterate through all transformer blocks
    for name, module in model.named_modules():
        # Check if this module should get LoRA
        should_apply = any(target in name for target in target_modules)

        if should_apply and isinstance(module, nn.Linear):
            # Get the parent module to replace the layer
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)

            # Create LoRA-wrapped version
            lora_layer = LinearWithLoRA(
                linear=module,
                rank=rank,
                alpha=alpha,
                dropout=0.0,
                enable_lora=True
            )

            # Replace the original module
            setattr(parent, child_name, lora_layer)

            # Count parameters
            if lora_layer.lora is not None:
                lora_params = sum(p.numel() for p in lora_layer.lora.parameters())
                total_lora_params += lora_params
                print(f"Applied LoRA to {name}: +{lora_params:,} trainable params")

    return total_lora_params


def get_lora_parameters(model: nn.Module):
    """
    Extract only the LoRA parameters from a model.
    Use this to create an optimizer that only trains LoRA weights.

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LinearWithLoRA) and module.lora is not None:
            lora_params.extend(module.lora.parameters())
    return lora_params


def merge_all_lora_weights(model: nn.Module):
    """
    Merge all LoRA weights into base weights for efficient inference.
    After merging, the model behaves identically but without LoRA overhead.
    """
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            module.merge_weights()
    print("All LoRA weights merged into base model")


def save_lora_weights(model: nn.Module, path: str):
    """
    Save only the LoRA adapter weights (not the full model).
    This creates very small checkpoint files.

    Args:
        model: Model with LoRA layers
        path: Path to save LoRA weights
    """
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA) and module.lora is not None:
            lora_state[name] = {
                'lora_A': module.lora.lora_A.data.clone(),
                'lora_B': module.lora.lora_B.data.clone(),
                'rank': module.lora.rank,
                'alpha': module.lora.alpha,
            }

    torch.save(lora_state, path)
    print(f"Saved LoRA weights to {path}")
    print(f"File size: {sum(v['lora_A'].numel() + v['lora_B'].numel() for v in lora_state.values()):,} parameters")


def load_lora_weights(model: nn.Module, path: str):
    """
    Load LoRA adapter weights into a model.
    The model must already have LoRA layers applied.

    Args:
        model: Model with LoRA layers
        path: Path to saved LoRA weights
    """
    lora_state = torch.load(path, map_location='cpu')

    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA) and module.lora is not None:
            if name in lora_state:
                module.lora.lora_A.data.copy_(lora_state[name]['lora_A'])
                module.lora.lora_B.data.copy_(lora_state[name]['lora_B'])

    print(f"Loaded LoRA weights from {path}")
