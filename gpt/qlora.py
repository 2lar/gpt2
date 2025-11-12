"""
qlora.py: QLoRA (Quantized Low-Rank Adaptation) implementation

QLoRA combines two techniques:
1. NF4 quantization: Compress base model weights to 4-bit (8x memory reduction)
2. LoRA: Add trainable low-rank adapters (only train 0.1-1% of parameters)

Key insight: We can quantize the FROZEN base weights to save memory, while keeping
the small LoRA adapters in full precision for training.

Memory breakdown (GPT-2 124M):
- Original model: 124M × 4 bytes = 496 MB
- QLoRA model: 124M × 0.5 bytes (quantized) + 0.3M × 2 bytes (LoRA) = 62 MB + 0.6 MB ≈ 63 MB
- Savings: ~8x reduction!

References:
- QLoRA paper: https://arxiv.org/abs/2305.14314
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, List

from .quantization import QuantizedLinear, NF4Quantizer
from .lora import LoRALayer


class QuantizedLinearWithLoRA(nn.Module):
    """
    The heart of QLoRA: a quantized frozen base layer + trainable LoRA adapter.

    Forward pass:
        output = QuantizedLinear(x) + LoRA(x)
        output = (W_quantized @ x) + (B @ A @ x) × (alpha / rank)

    Where:
    - W_quantized: Frozen 4-bit base weights (dequantized on-the-fly)
    - A, B: Trainable LoRA matrices (full precision)

    Memory efficiency:
    - W_quantized: 4 bits per parameter (~1/8 size of float32)
    - A: [rank, in_features] full precision
    - B: [out_features, rank] full precision
    - Total: Massive savings when rank << hidden_dim
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = True,
        block_size: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        # Quantized frozen base layer
        self.quantized_linear = QuantizedLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            block_size=block_size,
            dtype=dtype,
        )

        # Trainable LoRA adapter
        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
        )

        self.rank = rank
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: frozen quantized base + trainable LoRA.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            output = quantized_linear(x) + lora(x)
        """
        # Base output (frozen, quantized)
        base_output = self.quantized_linear(x)

        # LoRA adaptation (trainable, full precision)
        lora_output = self.lora(x)

        # Combine
        return base_output + lora_output

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        block_size: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Convert a Linear layer to QuantizedLinearWithLoRA.

        Args:
            linear: Original Linear layer
            rank: LoRA rank
            alpha: LoRA scaling factor
            block_size: Quantization block size
            dtype: Compute dtype

        Returns:
            QLoRA version of the layer
        """
        qlora_layer = QuantizedLinearWithLoRA(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
            block_size=block_size,
            dtype=dtype,
        )

        # Quantize the original weights
        qlora_layer.quantized_linear.quantize_weights(linear.weight.data)

        # Copy bias if present
        if linear.bias is not None:
            qlora_layer.quantized_linear.bias.data.copy_(linear.bias.data.to(dtype))

        return qlora_layer


def apply_qlora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    block_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """
    Apply QLoRA to a model by replacing Linear layers with QuantizedLinearWithLoRA.

    Args:
        model: Model to modify
        rank: LoRA rank
        alpha: LoRA alpha scaling
        target_modules: List of module names to target (e.g., ['c_attn', 'c_proj'])
                       If None, applies to ALL Linear layers
        block_size: Quantization block size (smaller = more accurate, larger = more memory efficient)
        dtype: Compute dtype (should be bfloat16 for best performance)

    Returns:
        Number of trainable LoRA parameters added
    """
    trainable_params = 0

    def _replace_linear(parent_module, name, target_names):
        nonlocal trainable_params

        for child_name, child_module in parent_module.named_children():
            if isinstance(child_module, nn.Linear):
                # Check if this is a target module
                if target_names is None or any(target in child_name for target in target_names):
                    # Replace with QLoRA version
                    qlora_layer = QuantizedLinearWithLoRA.from_linear(
                        child_module,
                        rank=rank,
                        alpha=alpha,
                        block_size=block_size,
                        dtype=dtype,
                    )

                    setattr(parent_module, child_name, qlora_layer)

                    # Count trainable params (only LoRA matrices)
                    trainable_params += qlora_layer.lora.lora_A.numel()
                    trainable_params += qlora_layer.lora.lora_B.numel()

                    print(f"  ✓ Converted {name}.{child_name} to QLoRA "
                          f"[{child_module.in_features} → {child_module.out_features}]")
            else:
                # Recursively process children
                _replace_linear(child_module, f"{name}.{child_name}", target_names)

    # Start replacement
    print(f"\nApplying QLoRA (rank={rank}, alpha={alpha}, block_size={block_size}):")
    _replace_linear(model, "model", target_modules)

    print(f"\nAdded {trainable_params:,} trainable LoRA parameters")
    return trainable_params


def get_qlora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get only the trainable LoRA parameters from a QLoRA model.

    Args:
        model: Model with QLoRA layers

    Returns:
        List of trainable LoRA parameters
    """
    lora_params = []

    for module in model.modules():
        if isinstance(module, QuantizedLinearWithLoRA):
            # Only LoRA matrices are trainable
            lora_params.append(module.lora.lora_A)
            lora_params.append(module.lora.lora_B)

    return lora_params


def save_qlora_weights(model: nn.Module, path: str):
    """
    Save only the LoRA adapter weights (not the quantized base).

    This saves a tiny checkpoint containing only the trainable adapters.
    The base model weights are already in the original checkpoint.

    Args:
        model: Model with QLoRA layers
        path: Path to save checkpoint
    """
    lora_state = {}

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinearWithLoRA):
            # Save LoRA matrices
            lora_state[f"{name}.lora_A"] = module.lora.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora.lora_B.data.cpu()

    torch.save(lora_state, path)
    print(f"Saved LoRA weights to {path}")


def load_qlora_weights(model: nn.Module, path: str):
    """
    Load LoRA adapter weights into a QLoRA model.

    Args:
        model: Model with QLoRA layers
        path: Path to LoRA checkpoint
    """
    lora_state = torch.load(path, map_location='cpu')

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinearWithLoRA):
            # Load LoRA matrices
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"

            if a_key in lora_state:
                module.lora.lora_A.data.copy_(lora_state[a_key])
            if b_key in lora_state:
                module.lora.lora_B.data.copy_(lora_state[b_key])

    print(f"Loaded LoRA weights from {path}")


def print_qlora_summary(model: nn.Module):
    """
    Print a detailed summary of QLoRA configuration and memory usage.

    Args:
        model: Model with QLoRA layers
    """
    total_params = 0
    trainable_params = 0
    quantized_params = 0

    for module in model.modules():
        if isinstance(module, QuantizedLinearWithLoRA):
            # Quantized base weights
            quantized_params += module.quantized_linear.weight_quantized.numel()

            # LoRA trainable params
            trainable_params += module.lora.lora_A.numel()
            trainable_params += module.lora.lora_B.numel()

        elif isinstance(module, nn.Linear):
            # Regular linear layers (not QLoRA'd)
            total_params += module.weight.numel()

    # Estimate memory
    # Quantized: 4 bits = 0.5 bytes per param
    # Trainable: bfloat16 = 2 bytes per param
    quantized_mb = (quantized_params * 0.5) / (1024 ** 2)
    trainable_mb = (trainable_params * 2) / (1024 ** 2)
    other_mb = (total_params * 4) / (1024 ** 2)  # Assume float32

    total_mb = quantized_mb + trainable_mb + other_mb

    print("\n" + "="*60)
    print("QLoRA Model Summary")
    print("="*60)
    print(f"Quantized parameters:  {quantized_params:,} ({quantized_mb:.2f} MB)")
    print(f"Trainable parameters:  {trainable_params:,} ({trainable_mb:.2f} MB)")
    print(f"Other parameters:      {total_params:,} ({other_mb:.2f} MB)")
    print(f"Total memory:          ~{total_mb:.2f} MB")
    print(f"Trainable ratio:       {trainable_params/(quantized_params+trainable_params+total_params)*100:.3f}%")
    print("="*60 + "\n")
