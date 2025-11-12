"""
finetune_qlora.py: Fine-tune GPT-2 using QLoRA (Quantized LoRA)

QLoRA = 4-bit quantized base model + LoRA adapters

This script demonstrates the EXTREME memory efficiency of QLoRA:
- Base GPT-2 (124M params): 496 MB → 62 MB (8x reduction via NF4 quantization)
- LoRA adapters: ~0.3M params at full precision
- Total: ~63 MB vs 496 MB for regular fine-tuning!

This allows fine-tuning large models on consumer GPUs (even 4GB VRAM can work!)

Usage:
    # Basic usage (works on 4GB VRAM!)
    python -m scripts.training.finetune_qlora --data brainrot_data --steps 500

    # Larger model (gpt2-medium, 350M params)
    python -m scripts.training.finetune_qlora \
      --model gpt2-medium \
      --data your_data \
      --steps 1000

Key differences from regular LoRA:
1. Base weights are quantized to 4-bit (frozen)
2. LoRA adapters stay in bfloat16 (trainable)
3. ~8x memory savings
4. Slight slowdown due to dequantization overhead

References:
- QLoRA paper: https://arxiv.org/abs/2305.14314
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from gpt.config import GPTConfig
from gpt.data import DataLoaderLite, LoaderConfig
from gpt.model import GPT
from gpt.qlora import (
    apply_qlora_to_model,
    get_qlora_parameters,
    save_qlora_weights,
    print_qlora_summary,
)


@dataclass
class QLoRAFineTuneConfig:
    """Configuration for QLoRA fine-tuning."""
    # Data
    data_root: str = "brainrot_data"

    # Model
    model_type: str = "gpt2"  # gpt2, gpt2-medium, gpt2-large, gpt2-xl
    weights_path: str = "data/gpt2_weights.pt"

    # QLoRA parameters
    lora_rank: int = 8
    lora_alpha: float = 16.0
    quantization_block_size: int = 64  # Smaller = more accurate, larger = more memory efficient

    # Training - can use larger batches than regular LoRA due to quantization!
    batch_size: int = 8  # 2x larger than regular LoRA (we have more memory!)
    seq_len: int = 256  # Can also use longer sequences
    grad_accum_steps: int = 4  # Effective batch size = 8 * 4 = 32
    max_steps: int = 500
    eval_interval: int = 50
    eval_iters: int = 1  # Single validation batch

    # Optimizer
    learning_rate: float = 3e-4  # Higher LR for LoRA
    weight_decay: float = 0.01
    warmup_steps: int = 50

    # Output
    output_dir: str = "qlora_checkpoints"
    log_interval: int = 10


def get_cosine_lr(step: int, cfg: QLoRAFineTuneConfig) -> float:
    """Cosine learning rate schedule with warmup."""
    # Warmup
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps

    # Cosine decay
    if step > cfg.max_steps:
        return cfg.learning_rate * 0.1

    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.1 + 0.5 * cfg.learning_rate * 0.9 * (1 + math.cos(math.pi * progress))


def evaluate_loss(model: nn.Module, val_loader: DataLoaderLite, eval_iters: int, device: str) -> float:
    """Evaluate validation loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu', dtype=torch.bfloat16):
                _, loss = model(x, y)

            total_loss += loss.item()

    model.train()
    return total_loss / eval_iters


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 with QLoRA")
    parser.add_argument("--data", type=str, default="brainrot_data", help="Data directory")
    parser.add_argument("--model", type=str, default="gpt2", help="Model size")
    parser.add_argument("--weights", type=str, default="data/gpt2_weights.pt", help="Pretrained weights")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--block-size", type=int, default=64, help="Quantization block size")
    parser.add_argument("--output", type=str, default="qlora_checkpoints", help="Output directory")

    args = parser.parse_args()

    # Create config
    cfg = QLoRAFineTuneConfig(
        data_root=args.data,
        model_type=args.model,
        weights_path=args.weights,
        max_steps=args.steps,
        lora_rank=args.rank,
        learning_rate=args.lr,
        quantization_block_size=args.block_size,
        output_dir=args.output,
    )

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("\n⚠️  WARNING: QLoRA is designed for GPU training.")
        print("   CPU training will be VERY slow due to dequantization overhead.")
        print("   Consider using regular LoRA for CPU training.\n")

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ========================================================================
    # 1. Load pretrained model
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Loading pretrained {cfg.model_type} model...")
    print(f"{'='*60}")

    model = GPT.from_pretrained(cfg.model_type, weights_path=cfg.weights_path)

    # Count base parameters BEFORE quantization
    base_params = sum(p.numel() for p in model.parameters())
    base_memory_mb = (base_params * 4) / (1024 ** 2)  # float32 = 4 bytes
    print(f"Base model parameters: {base_params/1e6:.2f}M ({base_memory_mb:.2f} MB)")

    # ========================================================================
    # 2. Apply QLoRA (quantization + LoRA adapters)
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Applying QLoRA...")
    print(f"  - Quantization: NF4 4-bit (block_size={cfg.quantization_block_size})")
    print(f"  - LoRA: rank={cfg.lora_rank}, alpha={cfg.lora_alpha}")
    print(f"{'='*60}")

    lora_params_count = apply_qlora_to_model(
        model,
        rank=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        target_modules=['c_attn'],  # Apply to Q, K, V projections
        block_size=cfg.quantization_block_size,
        dtype=torch.bfloat16,
    )

    # Move model to device
    model.to(device)

    # Print detailed memory summary
    print_qlora_summary(model)

    # Memory comparison
    quantized_memory_mb = (base_params * 0.5 + lora_params_count * 2) / (1024 ** 2)
    memory_savings = base_memory_mb - quantized_memory_mb
    compression_ratio = base_memory_mb / quantized_memory_mb

    print(f"Memory comparison:")
    print(f"  Original model:     {base_memory_mb:.2f} MB")
    print(f"  QLoRA model:        {quantized_memory_mb:.2f} MB")
    print(f"  Savings:            {memory_savings:.2f} MB ({compression_ratio:.1f}x reduction)")
    print(f"  Trainable ratio:    {lora_params_count/base_params*100:.3f}%")

    # ========================================================================
    # 3. Setup data loaders
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Setting up data loaders from {cfg.data_root}...")
    print(f"{'='*60}")

    train_loader = DataLoaderLite(LoaderConfig(
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        process_rank=0,
        world_size=1,
        split="train",
        data_root=cfg.data_root,
        master_process=True
    ))

    val_loader = DataLoaderLite(LoaderConfig(
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        process_rank=0,
        world_size=1,
        split="val",
        data_root=cfg.data_root,
        master_process=True
    ))

    # ========================================================================
    # 4. Setup optimizer (only LoRA parameters)
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Setting up optimizer (only LoRA parameters)...")
    print(f"{'='*60}")

    qlora_params = get_qlora_parameters(model)
    print(f"Trainable LoRA parameters: {sum(p.numel() for p in qlora_params):,}")

    optimizer = torch.optim.AdamW(
        qlora_params,
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    # ========================================================================
    # 5. Training loop
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Starting QLoRA fine-tuning for {cfg.max_steps} steps...")
    print(f"Effective batch size: {cfg.batch_size * cfg.grad_accum_steps}")
    print(f"{'='*60}\n")

    model.train()

    for step in range(cfg.max_steps):
        t0 = time.time()

        # Zero gradients
        optimizer.zero_grad()

        # Gradient accumulation
        loss_accum = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # Forward pass with mixed precision
            with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu', dtype=torch.bfloat16):
                _, loss = model(x, y)

            # Scale loss for gradient accumulation
            loss = loss / cfg.grad_accum_steps
            loss_accum += loss.detach()

            # Backward pass
            loss.backward()

        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(qlora_params, 1.0)

        # Update learning rate
        lr = get_cosine_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Optimizer step
        optimizer.step()

        # Timing
        t1 = time.time()
        dt = (t1 - t0) * 1000  # milliseconds
        tokens_per_sec = (cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps) / (t1 - t0)

        # Logging
        if step % cfg.log_interval == 0:
            print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.2e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.0f}")

        # Evaluation
        if step % cfg.eval_interval == 0 and step > 0:
            val_loss = evaluate_loss(model, val_loader, cfg.eval_iters, device)
            print(f"{'='*60}")
            print(f"step {step} | val_loss: {val_loss:.6f}")
            print(f"{'='*60}")

            # Save QLoRA checkpoint (only LoRA adapters, not quantized base)
            checkpoint_path = os.path.join(cfg.output_dir, f"qlora_step_{step:05d}.pt")
            save_qlora_weights(model, checkpoint_path)

    # ========================================================================
    # 6. Save final QLoRA weights
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")

    final_path = os.path.join(cfg.output_dir, "qlora_final.pt")
    save_qlora_weights(model, final_path)

    print(f"\nFinal QLoRA weights saved to: {final_path}")
    print(f"\nNote: This checkpoint contains ONLY the LoRA adapters.")
    print(f"To use it, you need:")
    print(f"  1. Base model weights: {cfg.weights_path}")
    print(f"  2. QLoRA adapters: {final_path}")
    print(f"\nUse scripts/inference/generate_qlora.py to generate text!")


if __name__ == "__main__":
    main()
