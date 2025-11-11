"""
finetune_lora.py: Fine-tune GPT-2 using LoRA on custom text

This script:
1. Loads a pretrained GPT-2 model
2. Applies LoRA adapters to attention layers (Q and V projections)
3. Trains only the LoRA parameters (base model stays frozen)
4. Uses gradient accumulation for 8GB VRAM compatibility
5. Saves only the small LoRA adapter weights

Usage:
    python -m scripts.finetune_lora --data brainrot_data --steps 500
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
from gpt.lora import apply_lora_to_model, get_lora_parameters, save_lora_weights


@dataclass
class LoRAFineTuneConfig:
    """Configuration for LoRA fine-tuning."""
    # Data
    data_root: str = "brainrot_data"

    # Model
    model_type: str = "gpt2"  # gpt2, gpt2-medium, gpt2-large, gpt2-xl
    weights_path: str = "data/gpt2_weights.pt"

    # LoRA
    lora_rank: int = 8
    lora_alpha: float = 16.0

    # Training - this part is adjustable depending on VRAM and data size
    batch_size: int = 4  # Small batch for 8GB VRAM
    seq_len: int = 128  # Shorter sequences for limited data
    grad_accum_steps: int = 4  # Effective batch size = 4 * 4 = 16
    max_steps: int = 500
    eval_interval: int = 50
    eval_iters: int = 1  # ADJUSTED: Single validation batch (for tiny datasets)

    # Optimizer
    learning_rate: float = 3e-4  # Higher LR for LoRA (adapters learn faster)
    weight_decay: float = 0.01  # Light weight decay
    warmup_steps: int = 50

    # Output
    output_dir: str = "lora_checkpoints"
    log_interval: int = 10


def get_cosine_lr(step: int, cfg: LoRAFineTuneConfig) -> float:
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
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 with LoRA")
    parser.add_argument("--data", type=str, default="brainrot_data", help="Data directory")
    parser.add_argument("--model", type=str, default="gpt2", help="Model size")
    parser.add_argument("--weights", type=str, default="data/gpt2_weights.pt", help="Pretrained weights")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default="lora_checkpoints", help="Output directory")

    args = parser.parse_args()

    # Create config
    cfg = LoRAFineTuneConfig(
        data_root=args.data,
        model_type=args.model,
        weights_path=args.weights,
        max_steps=args.steps,
        lora_rank=args.rank,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ========================================================================
    # 1. Load pretrained model
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Loading pretrained {cfg.model_type} model...")
    print(f"{'='*60}")

    model = GPT.from_pretrained(cfg.model_type, weights_path=cfg.weights_path)
    model.to(device)

    # Count base parameters
    base_params = sum(p.numel() for p in model.parameters())
    print(f"Base model parameters: {base_params/1e6:.2f}M")

    # ========================================================================
    # 2. Apply LoRA to attention layers
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Applying LoRA (rank={cfg.lora_rank}, alpha={cfg.lora_alpha})...")
    print(f"{'='*60}")

    lora_params_count = apply_lora_to_model(
        model,
        rank=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        target_modules=['c_attn']  # Apply to Q, K, V projections
    )

    # Move LoRA parameters to device (they're created on CPU by default)
    model.to(device)

    print(f"\nLoRA parameters: {lora_params_count/1e6:.2f}M")
    print(f"Trainable ratio: {lora_params_count/base_params*100:.2f}%")

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

    lora_params = get_lora_parameters(model)
    print(f"Trainable LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    # ========================================================================
    # 5. Training loop
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Starting LoRA fine-tuning for {cfg.max_steps} steps...")
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
        norm = torch.nn.utils.clip_grad_norm_(lora_params, 1.0)

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

            # Save LoRA checkpoint
            checkpoint_path = os.path.join(cfg.output_dir, f"lora_step_{step:05d}.pt")
            save_lora_weights(model, checkpoint_path)

    # ========================================================================
    # 6. Save final LoRA weights
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")

    final_path = os.path.join(cfg.output_dir, "lora_final.pt")
    save_lora_weights(model, final_path)

    print(f"\nFinal LoRA weights saved to: {final_path}")
    print(f"Use scripts/compare_outputs.py to see before/after results!")


if __name__ == "__main__":
    main()
