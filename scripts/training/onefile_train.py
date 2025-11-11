"""
train.py: Training entry point that mirrors the reference example while keeping
modules separated.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt.config import GPTConfig
from gpt.data import DataLoaderLite, LoaderConfig
from gpt.evaluation import get_most_likely_row
from gpt.model import GPT

try:
    import tiktoken
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError("tiktoken is required for training") from exc

try:
    # HellaSwag script is in the same directory (scripts/)
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from eval_hellaswag import iterate_examples, render_example
except (ModuleNotFoundError, ImportError):
    iterate_examples = render_example = None


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    # Data configuration
    data_root: str = "shakespeare_data"  # Directory containing train/val .npy shards

    # Batch size configuration
    # total_batch_size = total tokens processed before weight update
    # This is split across: micro_batch_size × grad_accum_steps × num_gpus
    total_batch_size: int = 32768  # Chosen to match GPT-2's token-per-step target so optimization dynamics resemble the paper.
    micro_batch_size: int = 16     # Small enough to fit activations on consumer GPUs; grad accumulation rebuilds the large global batch.
    seq_len: int = 256             # Short context keeps experiments cheap while still exercising positional modeling.

    # Evaluation configuration
    eval_interval: int = 50   # Frequent evals catch regressions quickly when iterating on ablations.
    eval_iters: int = 10      # Averaging a few batches smooths noise without burning lots of tokens on validation.

    # Learning rate configuration
    max_lr: float = 6e-4       # Proven sweet spot for GPT-2-scale AdamW training; too high blows up, too low trains slowly.
    min_lr_ratio: float = 0.1  # Ending at 10% of max preserves progress late in cosine decay instead of collapsing LR to zero.
    warmup_steps: int = 100    # Short warmup prevents early gradient spikes while keeping ramp-up overhead minimal.

    # Training duration
    max_steps: int = 1000  # Enough steps to see trends without the cost of a full epoch on the dataset.

    # Logging and checkpointing
    log_dir: str = "log"  # Single place for metrics/checkpoints simplifies experiment bookkeeping.

    # Compilation (experimental PyTorch 2.0 feature)
    compile: bool = False  # Disabled by default; debugging compiled graphs is harder, so opt-in once runs are stable.

    # Evaluation benchmarks
    hellaswag_interval: int = 250  # Expensive benchmark, so run sparsely to keep wall-clock reasonable.

    # Text generation sampling
    sampling_interval: int = 250  # Text samples are qualitative checks; periodic snapshots avoid slowing the loop.
    sample_prompt: str = "Hello, I'm a language model,"  # Neutral prompt that exposes general language ability.
    sample_max_len: int = 32      # Short completions minimize time spent in slow autoregressive loops.
    sample_count: int = 4         # Few samples balance diversity with logging verbosity.

    # Optimizer configuration
    weight_decay: float = 0.1  # Same decay OpenAI reported; discourages overfitting without hampering large batches.


def setup_distributed() -> Tuple[bool, int, int, int, str, str, bool]:
    """
    Setup distributed training (multi-GPU) if environment variables are set.

    Uses the modern torch.accelerator API (PyTorch 2.8+) for device-agnostic code.
    Falls back to torch.cuda for older PyTorch versions.

    Returns:
        ddp: Whether DDP is enabled
        ddp_rank: Global rank of this process (0 to world_size-1)
        ddp_local_rank: Local rank on this machine (0 to num_gpus_per_machine-1)
        ddp_world_size: Total number of processes across all machines
        device: Device string (e.g., 'cuda:0', 'cuda', 'cpu')
        device_type: Device type ('cuda' or 'cpu')
        master_process: Whether this is rank 0 (main process for logging)
    """
    # Check if running with torchrun (Distributed Data Parallel)
    # torchrun sets RANK environment variable
    ddp = int(os.environ.get('RANK', -1)) != -1

    # Check if modern accelerator API is available (PyTorch 2.8+)
    has_accelerator_api = hasattr(torch, 'accelerator')

    if ddp:
        # Multi-GPU distributed training mode
        if has_accelerator_api:
            assert torch.accelerator.is_available(), "DDP run requires an accelerator (GPU)"
        else:
            assert torch.cuda.is_available(), "DDP run requires CUDA"

        # Initialize process group for communication between GPUs
        # 'nccl' backend is optimized for NVIDIA GPUs (also supports AMD ROCm)
        dist.init_process_group(backend='nccl')

        # Get distributed training info from environment (set by torchrun)
        ddp_rank = int(os.environ['RANK'])              # Global rank (0, 1, 2, ...)
        ddp_local_rank = int(os.environ['LOCAL_RANK'])  # Local GPU index (0, 1, 2, ...)
        ddp_world_size = int(os.environ['WORLD_SIZE'])  # Total number of GPUs

        # Each process gets its own GPU based on LOCAL_RANK
        # Modern API (PyTorch 2.8+): torch.accelerator.set_device_index()
        # Legacy API: torch.cuda.set_device()
        if has_accelerator_api:
            device = f'{torch.accelerator.current_accelerator()}:{ddp_local_rank}'
            torch.accelerator.set_device_index(ddp_local_rank)
        else:
            device = f'cuda:{ddp_local_rank}'
            torch.cuda.set_device(device)

        # Only rank 0 should print logs and save checkpoints
        master_process = ddp_rank == 0
    else:
        # Single GPU or CPU training mode
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        # Auto-detect best available device
        # Modern API: Use torch.accelerator if available
        if has_accelerator_api and torch.accelerator.is_available():
            # PyTorch 2.8+ device-agnostic API
            device = torch.accelerator.current_accelerator()
        else:
            # Legacy device detection (works on all PyTorch versions)
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU

        print(f"using device: {device}")

    # Device type is used for autocast (mixed precision)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, str(device), device_type, master_process


def teardown_distributed(enabled: bool) -> None:
    """Clean up distributed training resources."""
    if enabled:
        dist.destroy_process_group()


def get_lr(step: int, cfg: TrainingConfig) -> float:
    """
    Calculate learning rate for given step using warmup + cosine decay schedule.

    Schedule:
    1. Linear warmup: 0 → max_lr over warmup_steps
    2. Cosine decay: max_lr → min_lr over remaining steps

    This is a common schedule that:
    - Prevents unstable training at start (warmup)
    - Gradually reduces LR for fine-grained optimization (decay)

    Args:
        step: Current training step
        cfg: Training configuration with max_lr, min_lr_ratio, warmup_steps, max_steps

    Returns:
        Learning rate for this step
    """
    # Calculate minimum learning rate
    min_lr = cfg.max_lr * cfg.min_lr_ratio  # e.g., 6e-4 * 0.1 = 6e-5

    # Phase 1: Linear warmup (steps 0 to warmup_steps)
    # LR increases linearly from 0 to max_lr
    # This prevents large gradient updates when weights are random
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps

    # After max_steps: stay at minimum LR
    if step > cfg.max_steps:
        return min_lr

    # Phase 2: Cosine decay (steps warmup_steps to max_steps)
    # LR decreases smoothly from max_lr to min_lr following cosine curve
    # Calculate progress through decay phase (0.0 to 1.0)
    decay_ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)  # Clamp to [0, 1]

    # Cosine coefficient: starts at 1.0, ends at 0.0
    # cos(0) = 1, cos(π) = -1, so (1 + cos(π * progress)) / 2 goes from 1 to 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    # Interpolate between min_lr and max_lr using cosine coefficient
    return min_lr + coeff * (cfg.max_lr - min_lr)


def build_dataloaders(cfg: TrainingConfig, ddp_rank: int, ddp_world_size: int,
                      master_process: bool) -> Tuple[DataLoaderLite, DataLoaderLite]:
    common = dict(
        batch_size=cfg.micro_batch_size,
        seq_len=cfg.seq_len,
        process_rank=ddp_rank,
        world_size=ddp_world_size,
        data_root=cfg.data_root,
        master_process=master_process,
    )
    train_loader = DataLoaderLite(LoaderConfig(split="train", **common))
    val_loader = DataLoaderLite(LoaderConfig(split="val", **common))
    return train_loader, val_loader


def maybe_prepare_hellaswag(master_process: bool) -> bool:
    if iterate_examples is None or render_example is None:
        if master_process:
            print("hellaswag package not available; skipping hella evaluation")
        return False
    return True


def train(model_cfg: Optional[GPTConfig] = None, train_cfg: Optional[TrainingConfig] = None) -> None:
    """Main training function."""

    # ========================================================================
    # 1. Configuration Setup
    # ========================================================================

    # Use default configs if not provided
    # vocab_size=50304 is optimized for GPU efficiency (divisible by 64)
    model_cfg = model_cfg or GPTConfig(vocab_size=50304)
    train_cfg = train_cfg or TrainingConfig()

    # Setup distributed training (multi-GPU) if running with torchrun
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, master_process = setup_distributed()

    # Set random seed for reproducibility
    # Same seed = same random initialization = reproducible results
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Load tokenizer (converts text <-> token IDs)
    # GPT-2 uses BPE (Byte Pair Encoding) with 50,257 tokens
    enc = tiktoken.get_encoding("gpt2")

    # ========================================================================
    # 2. Calculate Gradient Accumulation Steps
    # ========================================================================
    # How many micro-batches to accumulate before updating weights?
    #
    # Formula: total_batch_size = micro_batch_size × seq_len × grad_accum_steps × num_gpus
    # Solving for grad_accum_steps:
    #   grad_accum_steps = total_batch_size / (micro_batch_size × seq_len × num_gpus)
    #
    # Example: total_batch_size=32768, micro_batch_size=16, seq_len=256, num_gpus=1
    #   grad_accum_steps = 32768 / (16 × 256 × 1) = 32768 / 4096 = 8
    grad_accum_steps = train_cfg.total_batch_size // (train_cfg.micro_batch_size * train_cfg.seq_len * ddp_world_size)

    # Validate that batch size is evenly divisible
    if grad_accum_steps == 0 or train_cfg.total_batch_size % (train_cfg.micro_batch_size * train_cfg.seq_len * ddp_world_size) != 0:
        raise ValueError("total_batch_size must be divisible by micro_batch_size * seq_len * world_size")

    if master_process:
        print(f"total desired batch size: {train_cfg.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # ========================================================================
    # 3. Setup Data Loaders
    # ========================================================================
    # Create train and validation data loaders
    # These load .npy shards and yield (input, target) token batches
    train_loader, val_loader = build_dataloaders(train_cfg, ddp_rank, ddp_world_size, master_process)

    # ========================================================================
    # 4. Create Model
    # ========================================================================
    # GLOBAL SETTING: Use TensorFloat32 (TF32) for faster matmuls on Ampere+ GPUs
    # This affects ALL float32 matrix multiplications in the entire program
    # - 'highest' = full float32 (slowest, most accurate)
    # - 'high'    = TF32 (8x faster on Ampere+, negligible accuracy loss)
    # - 'medium'  = more aggressive (fastest, slight accuracy loss)
    # Only works on: RTX 30xx/40xx, A100, H100, etc. (ignored on older GPUs)
    # Does NOT affect bfloat16/float16 ops (we use those in autocast anyway)
    torch.set_float32_matmul_precision('high')

    # Create GPT model with random weights (training from scratch)
    model = GPT(model_cfg)
    model.to(device)  # Move to GPU/CPU

    # Optional: Compile model with PyTorch 2.0 for faster training
    if train_cfg.compile:
        print("compiling model with torch.compile() (this may take a while)...")
        model = torch.compile(model)

    # Wrap in DistributedDataParallel for multi-GPU training
    # DDP synchronizes gradients across GPUs and averages them
    if ddp:
        print("wrapping model in DistributedDataParallel...")
        model = DDP(model, device_ids=[ddp_local_rank])

    # Get unwrapped model (needed for configure_optimizers)
    # DDP wraps the model, so we need .module to access the original
    raw_model = model.module if ddp else model

    # ========================================================================
    # 5. Create Optimizer
    # ========================================================================
    # AdamW optimizer with weight decay (L2 regularization)
    # Configured in model.configure_optimizers() with:
    # - Weight decay only on 2D parameters (not biases/LayerNorm)
    # - Fused kernels for faster updates on CUDA
    optimizer = raw_model.configure_optimizers(
        weight_decay=train_cfg.weight_decay,
        learning_rate=train_cfg.max_lr,  # Initial LR (will be updated each step)
        device_type=device_type,
        master_process=master_process,
    )

    # ========================================================================
    # 6. Setup Logging
    # ========================================================================
    # Create log directory and empty log file
    os.makedirs(train_cfg.log_dir, exist_ok=True)
    log_file = os.path.join(train_cfg.log_dir, "log.txt")
    with open(log_file, "w"):
        pass  # Clear file

    # Prepare HellaSwag benchmark (if available)
    hellaswag_enabled = maybe_prepare_hellaswag(master_process)

    # ========================================================================
    # 7. Training Loop
    # ========================================================================
    for step in range(train_cfg.max_steps):
        t0 = time.time()
        last_step = step == train_cfg.max_steps - 1

        # Validation monitoring - run on validation set periodically
        if step % train_cfg.eval_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(train_cfg.eval_iters):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    loss = loss / train_cfg.eval_iters
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    checkpoint_path = os.path.join(train_cfg.log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                    }
                    torch.save(checkpoint, checkpoint_path)

        # HellaSwag evaluation - run full benchmark periodically
        if (step % train_cfg.hellaswag_interval == 0 or last_step) and hellaswag_enabled and not train_cfg.compile:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # Text generation sampling - generate samples periodically
        if ((step > 0 and step % train_cfg.sampling_interval == 0) or last_step) and not train_cfg.compile:
            model.eval()
            tokens = enc.encode(train_cfg.sample_prompt)
            tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            tokens = tokens.repeat(train_cfg.sample_count, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < train_cfg.sample_max_len:
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = model(xgen)
                    logits = logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
            for i in range(train_cfg.sample_count):
                tokens = xgen[i, :train_cfg.sample_max_len].tolist()
                decoded = enc.decode(tokens)
                if master_process:
                    print(f"rank {ddp_rank} sample {i}: {decoded}")

        # ====================================================================
        # MAIN TRAINING STEP - This is where learning happens!
        # ====================================================================

        model.train()  # Set model to training mode (enables dropout, etc.)

        # Step 1: Zero out gradients from previous step
        # Without this, gradients would accumulate across steps (we don't want that)
        optimizer.zero_grad()

        # Step 2: Gradient Accumulation Loop
        # We simulate a large batch by accumulating gradients over multiple small batches
        # Example: Instead of 1 batch of 32768 tokens (too big for GPU),
        #          we do 2048 batches of 16 tokens and accumulate their gradients
        loss_accum = 0.0  # Track total loss across micro-batches

        for micro_step in range(grad_accum_steps):
            # Get next batch of data
            # x: input tokens (B, T) - what the model sees
            # y: target tokens (B, T) - what the model should predict
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)  # Move to GPU

            # DDP optimization: Only sync gradients on the last micro-step
            # This avoids expensive communication between GPUs until we're ready to update
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            # Forward pass with mixed precision (bfloat16 for speed)
            # model(x, y) returns (logits, loss)
            # logits: predictions for each token (B, T, vocab_size)
            # loss: cross-entropy loss (scalar)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)

            # Scale loss by number of accumulation steps
            # Why? We're averaging gradients across micro-batches
            # Without this, gradients would be grad_accum_steps times too large
            loss = loss / grad_accum_steps

            # Accumulate loss for logging (detach to avoid keeping computation graph)
            loss_accum += loss.detach()

            # Backward pass: Compute gradients
            # This calculates ∂loss/∂weight for every parameter in the model
            # Gradients are accumulated (added) to existing gradients from previous micro-steps
            loss.backward()

        # Step 3: Average loss across all GPUs (if using DDP)
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Step 4: Gradient Clipping
        # Prevents "exploding gradients" by limiting gradient magnitude to 1.0
        # Returns the global norm of gradients (for monitoring)
        # If norm > 1.0, all gradients are scaled down proportionally
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Step 5: Update Learning Rate
        # Learning rate changes each step according to schedule (warmup + cosine decay)
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Step 6: Optimizer Step - UPDATE THE WEIGHTS!
        # This is where learning actually happens
        # For each parameter: W_new = W_old - lr * gradient
        # AdamW uses momentum and adaptive learning rates, but conceptually it's the same
        optimizer.step()

        # Step 7: Timing and Logging
        # Synchronize GPU to get accurate timing
        if device_type == "cuda":
            torch.cuda.synchronize()

        # Calculate how long this step took
        dt = time.time() - t0

        # Calculate throughput (tokens processed per second)
        # B = batch size, T = sequence length
        # We process B*T tokens per micro-step, grad_accum_steps micro-steps total
        # Multiply by ddp_world_size if using multiple GPUs
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        # Print training progress (only on main process to avoid spam)
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    teardown_distributed(ddp)


if __name__ == "__main__":
    train()
