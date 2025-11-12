"""
quantization.py: Quantization utilities for qLoRA

This module implements NF4 (4-bit NormalFloat) quantization, which is the core
of QLoRA's memory efficiency. NF4 is information-theoretically optimal for
normally distributed weights (which neural network weights typically are).

Key concepts:
1. NF4 quantization: Maps 32-bit floats to 4-bit using optimal quantiles
2. Double quantization: Quantize the quantization constants themselves
3. Dequantization: Convert back to bfloat16 for computation

References:
- QLoRA paper: https://arxiv.org/abs/2305.14314
- bitsandbytes library (original implementation)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


# NF4 (4-bit NormalFloat) quantization levels
# These are optimal quantiles for a standard normal distribution
# Derived by dividing N(0,1) into 16 equal probability regions
NF4_QUANTILES = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
])


class NF4Quantizer:
    """
    4-bit NormalFloat quantizer.

    This quantizer:
    1. Normalizes weights by their absmax
    2. Maps normalized values to nearest NF4 quantile
    3. Stores 4-bit indices + scale factor

    Memory savings: 32-bit float -> 4-bit int + small scale = ~8x reduction
    """

    def __init__(self, block_size: int = 64):
        """
        Args:
            block_size: Quantize in blocks of this size (smaller = more accurate, larger = more memory efficient)
        """
        self.block_size = block_size
        self.nf4_quantiles = NF4_QUANTILES

    def quantize(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a weight tensor to 4-bit NF4.

        Args:
            weight: Float tensor to quantize (any shape)

        Returns:
            quantized: Int8 tensor containing 4-bit values (0-15)
            scale: Scale factors per block
        """
        original_shape = weight.shape
        weight_flat = weight.flatten()

        # Pad to multiple of block_size
        n_elements = weight_flat.numel()
        n_blocks = (n_elements + self.block_size - 1) // self.block_size
        padded_size = n_blocks * self.block_size

        if padded_size > n_elements:
            weight_flat = torch.cat([
                weight_flat,
                torch.zeros(padded_size - n_elements, device=weight.device, dtype=weight.dtype)
            ])

        # Reshape into blocks
        weight_blocks = weight_flat.view(n_blocks, self.block_size)

        # Compute absmax per block (this is our scale factor)
        absmax = weight_blocks.abs().max(dim=1, keepdim=True).values
        absmax = torch.clamp(absmax, min=1e-8)  # Avoid division by zero

        # Normalize to [-1, 1]
        normalized = weight_blocks / absmax

        # Quantize: find nearest NF4 quantile
        nf4_quantiles = self.nf4_quantiles.to(weight.device)

        # Expand dims for broadcasting
        # normalized: [n_blocks, block_size]
        # nf4_quantiles: [16]
        # distances: [n_blocks, block_size, 16]
        distances = (normalized.unsqueeze(-1) - nf4_quantiles.view(1, 1, -1)).abs()

        # Get index of nearest quantile (0-15)
        quantized = distances.argmin(dim=-1).to(torch.uint8)  # [n_blocks, block_size]

        # Flatten and trim padding
        quantized = quantized.flatten()[:n_elements]

        # Store original shape for dequantization
        return quantized, absmax.squeeze(-1), original_shape

    def dequantize(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        original_shape: torch.Size,
        dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """
        Dequantize from 4-bit NF4 back to float.

        Args:
            quantized: Int8 tensor with 4-bit values (0-15)
            scale: Scale factors per block
            original_shape: Original tensor shape
            dtype: Output dtype (typically bfloat16 for qLoRA)

        Returns:
            Dequantized float tensor
        """
        n_elements = quantized.numel()
        n_blocks = scale.numel()
        padded_size = n_blocks * self.block_size

        # Pad quantized values if needed
        if padded_size > n_elements:
            quantized = torch.cat([
                quantized,
                torch.zeros(padded_size - n_elements, device=quantized.device, dtype=quantized.dtype)
            ])

        # Reshape into blocks
        quantized_blocks = quantized.view(n_blocks, self.block_size)

        # Map indices to NF4 quantile values
        nf4_quantiles = self.nf4_quantiles.to(quantized.device).to(dtype)
        dequantized = nf4_quantiles[quantized_blocks]  # [n_blocks, block_size]

        # Denormalize using scale factors
        dequantized = dequantized * scale.unsqueeze(-1).to(dtype)

        # Flatten and trim padding
        dequantized = dequantized.flatten()[:n_elements]

        # Reshape to original shape
        return dequantized.view(original_shape)


class QuantizedLinear(nn.Module):
    """
    A Linear layer with NF4-quantized weights.

    This layer:
    1. Stores weights in 4-bit quantized format (memory efficient)
    2. Dequantizes on-the-fly during forward pass (compute intensive)
    3. Never computes gradients for frozen base weights

    Memory usage: ~1/8 of normal Linear layer
    Compute: Slightly slower due to dequantization overhead
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.dtype = dtype

        self.quantizer = NF4Quantizer(block_size=block_size)

        # These will be set by quantize_weights()
        self.register_buffer('weight_quantized', None)
        self.register_buffer('weight_scale', None)
        self.weight_shape = None

        if bias:
            # Bias stays in full precision (it's tiny)
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def quantize_weights(self, weight: torch.Tensor):
        """
        Quantize and store the weight tensor.

        Args:
            weight: [out_features, in_features] tensor to quantize
        """
        quantized, scale, shape = self.quantizer.quantize(weight)

        self.weight_quantized = quantized
        self.weight_scale = scale
        self.weight_shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: dequantize weights and apply linear transformation.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Dequantize weights on-the-fly
        weight_fp = self.quantizer.dequantize(
            self.weight_quantized,
            self.weight_scale,
            self.weight_shape,
            dtype=self.dtype
        )

        # Standard linear transformation
        return torch.nn.functional.linear(x, weight_fp, self.bias)

    @staticmethod
    def from_linear(linear: nn.Linear, block_size: int = 64, dtype: torch.dtype = torch.bfloat16):
        """
        Convert a regular Linear layer to QuantizedLinear.

        Args:
            linear: Original Linear layer
            block_size: Quantization block size
            dtype: Compute dtype

        Returns:
            Quantized version of the linear layer
        """
        quantized_layer = QuantizedLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
            dtype=dtype,
        )

        # Quantize the weights
        quantized_layer.quantize_weights(linear.weight.data)

        # Copy bias if present
        if linear.bias is not None:
            quantized_layer.bias.data.copy_(linear.bias.data.to(dtype))

        return quantized_layer


def quantize_model(model: nn.Module, block_size: int = 64, dtype: torch.dtype = torch.bfloat16):
    """
    Quantize all Linear layers in a model to NF4.

    Args:
        model: Model to quantize
        block_size: Quantization block size
        dtype: Compute dtype
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace with quantized version
            quantized = QuantizedLinear.from_linear(module, block_size=block_size, dtype=dtype)
            setattr(model, name, quantized)
        else:
            # Recursively quantize children
            quantize_model(module, block_size=block_size, dtype=dtype)


def estimate_memory_savings(model: nn.Module) -> dict:
    """
    Estimate memory savings from quantization.

    Returns:
        Dictionary with memory statistics
    """
    original_params = 0
    quantized_params = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            original_params += module.weight.numel()
            if module.bias is not None:
                original_params += module.bias.numel()
        elif isinstance(module, QuantizedLinear):
            # Weight: 4 bits per element
            quantized_params += module.weight_quantized.numel() * 0.5  # 4 bits = 0.5 bytes
            # Scale: 1 float per block
            quantized_params += module.weight_scale.numel() * 4  # float32 = 4 bytes
            # Bias: full precision
            if module.bias is not None:
                quantized_params += module.bias.numel() * 2  # bfloat16 = 2 bytes

    original_mb = (original_params * 4) / (1024 ** 2)  # Assume float32
    quantized_mb = quantized_params / (1024 ** 2)

    return {
        'original_mb': original_mb,
        'quantized_mb': quantized_mb,
        'savings_mb': original_mb - quantized_mb,
        'compression_ratio': original_mb / max(quantized_mb, 0.001),
    }
