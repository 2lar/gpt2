"""
train.py: Modular Training Script with Proper Separation of Concerns

This is the refactored version that demonstrates best practices:
- Extracted utilities into separate modules
- Clear separation of concerns
- Reusable components
- Professional code organization

For the educational all-in-one version, see onefile_train.py
"""
from __future__ import annotations

import os
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt.config import GPTConfig
from gpt.model import GPT

# Import modular training utilities (includes TrainingConfig)
from .lib import (
    TrainingConfig,
    setup_distributed,
    teardown_distributed,
    get_lr,
    evaluate_loss,
    run_hellaswag_eval,
    generate_samples,
    build_dataloaders,
)

try:
    import tiktoken
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("tiktoken is required for training") from exc

try:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ..evaluation.eval_hellaswag import iterate_examples, render_example
    from gpt.evaluation import get_most_likely_row
except (ModuleNotFoundError, ImportError):
    iterate_examples = render_example = get_most_likely_row = None


def train(model_cfg: Optional[GPTConfig] = None, train_cfg: Optional[TrainingConfig] = None) -> None:
    """Main training function - clean and modular."""

    # ========================================================================
    # Configuration Setup
    # ========================================================================
    model_cfg = model_cfg or GPTConfig(vocab_size=50304)
    train_cfg = train_cfg or TrainingConfig()

    # Setup distributed training
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, master_process = setup_distributed()

    # Set random seed
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # ========================================================================
    # Calculate Gradient Accumulation
    # ========================================================================
    grad_accum_steps = train_cfg.total_batch_size // (train_cfg.micro_batch_size * train_cfg.seq_len * ddp_world_size)

    if grad_accum_steps == 0 or train_cfg.total_batch_size % (train_cfg.micro_batch_size * train_cfg.seq_len * ddp_world_size) != 0:
        raise ValueError("total_batch_size must be divisible by micro_batch_size * seq_len * world_size")

    if master_process:
        print(f"total desired batch size: {train_cfg.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # ========================================================================
    # Setup Data Loaders
    # ========================================================================
    train_loader, val_loader = build_dataloaders(
        batch_size=train_cfg.micro_batch_size,
        seq_len=train_cfg.seq_len,
        data_root=train_cfg.data_root,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        master_process=master_process,
    )

    # ========================================================================
    # Create Model
    # ========================================================================
    torch.set_float32_matmul_precision('high')

    model = GPT(model_cfg)
    model.to(device)

    if train_cfg.compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    # ========================================================================
    # Create Optimizer
    # ========================================================================
    optimizer = raw_model.configure_optimizers(
        weight_decay=train_cfg.weight_decay,
        learning_rate=train_cfg.max_lr,
        device_type=device_type,
        master_process=master_process,
    )

    # ========================================================================
    # Setup Logging
    # ========================================================================
    os.makedirs(train_cfg.log_dir, exist_ok=True)
    log_file = os.path.join(train_cfg.log_dir, "log.txt")
    with open(log_file, "w"):
        pass

    hellaswag_enabled = iterate_examples is not None and render_example is not None

    # ========================================================================
    # Training Loop
    # ========================================================================
    for step in range(train_cfg.max_steps):
        t0 = time.time()
        last_step = step == train_cfg.max_steps - 1

        # Validation
        if step % train_cfg.eval_interval == 0 or last_step:
            model.eval()
            val_loader.reset()

            val_loss = evaluate_loss(model, val_loader, train_cfg.eval_iters, device, device_type)

            if ddp:
                val_loss_tensor = torch.tensor(val_loss, device=device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                val_loss = val_loss_tensor.item()

            if master_process:
                print(f"validation loss: {val_loss:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss:.4f}\n")

                if step > 0 and (step % 5000 == 0 or last_step):
                    checkpoint_path = os.path.join(train_cfg.log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss,
                    }
                    torch.save(checkpoint, checkpoint_path)

        # HellaSwag evaluation
        if (step % train_cfg.hellaswag_interval == 0 or last_step) and hellaswag_enabled and not train_cfg.compile:
            num_correct, num_total = run_hellaswag_eval(
                model, iterate_examples, render_example, get_most_likely_row,
                device, device_type, ddp_rank, ddp_world_size
            )

            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct = torch.tensor(num_correct, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct = num_correct.item()

            acc = num_correct / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct}/{num_total}={acc:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc:.4f}\n")

        # Text generation sampling
        if ((step > 0 and step % train_cfg.sampling_interval == 0) or last_step) and not train_cfg.compile:
            samples = generate_samples(
                model, enc, train_cfg.sample_prompt, train_cfg.sample_count,
                train_cfg.sample_max_len, device, device_type, ddp_rank
            )

            if master_process:
                for i, sample in enumerate(samples):
                    print(f"rank {ddp_rank} sample {i}: {sample}")

        # Training step
        model.train()
        optimizer.zero_grad()

        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        dt = time.time() - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    teardown_distributed(ddp)


if __name__ == "__main__":
    train()
