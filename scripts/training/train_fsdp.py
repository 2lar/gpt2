"""
train_fsdp.py: Train GPT-2 from scratch using FSDP2 (Fully Sharded Data Parallel v2)

FSDP2 is PyTorch's modern distributed training framework that shards model parameters,
gradients, and optimizer states across multiple GPUs to enable training models that
wouldn't fit on a single GPU.

=============================================================================
WHAT IS FSDP2 AND WHY USE IT?
=============================================================================

FSDP (Fully Sharded Data Parallel) shards the model across GPUs instead of
replicating it like DDP (DistributedDataParallel):

┌─────────────────────────────────────────────────────────────────┐
│ DDP (DistributedDataParallel):                                  │
│ GPU 0: [Full Model] [Full Optimizer State]                      │
│ GPU 1: [Full Model] [Full Optimizer State]                      │
│ GPU 2: [Full Model] [Full Optimizer State]                      │
│ GPU 3: [Full Model] [Full Optimizer State]                      │
│                                                                  │
│ - Each GPU holds a COMPLETE copy of the model                   │
│ - Gradients are all-reduced across GPUs                         │
│ - Memory: 4x model size (one per GPU)                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ FSDP2 (Fully Sharded Data Parallel v2):                         │
│ GPU 0: [Params 0-24M] [Grads 0-24M] [Optimizer State 0-24M]     │
│ GPU 1: [Params 25-49M] [Grads 25-49M] [Optimizer State 25-49M]  │
│ GPU 2: [Params 50-74M] [Grads 50-74M] [Optimizer State 50-74M]  │
│ GPU 3: [Params 75-99M] [Grads 75-99M] [Optimizer State 75-99M]  │
│                                                                  │
│ - Each GPU holds only 1/4 of the model parameters               │
│ - Parameters are gathered (all-gathered) only when needed       │
│ - After computation, gathered params are freed                  │
│ - Memory: 1x model size total (shared across GPUs)              │
└─────────────────────────────────────────────────────────────────┘

FSDP2 vs FSDP1 (the older API):
- FSDP1: Used FlatParameter (flattened param buffers), wrapper class
- FSDP2: Uses DTensor (distributed tensor), functional API with fully_shard()
- FSDP2 is cleaner, more composable, and the future of PyTorch distributed

=============================================================================
LAUNCH COMMAND
=============================================================================

Single node, 4 GPUs:
    torchrun --nproc_per_node=4 scripts/training/train_fsdp.py

Single node, all available GPUs:
    torchrun --nproc_per_node=auto scripts/training/train_fsdp.py

With custom arguments:
    torchrun --nproc_per_node=4 scripts/training/train_fsdp.py \
      --batch-size 8 \
      --seq-len 256 \
      --max-steps 1000

Multi-node (2 nodes, 4 GPUs each):
    # Node 0 (master):
    torchrun --nproc_per_node=4 \
      --nnodes=2 \
      --node_rank=0 \
      --master_addr=192.168.1.1 \
      --master_port=29500 \
      scripts/training/train_fsdp.py

    # Node 1:
    torchrun --nproc_per_node=4 \
      --nnodes=2 \
      --node_rank=1 \
      --master_addr=192.168.1.1 \
      --master_port=29500 \
      scripts/training/train_fsdp.py

=============================================================================
KEY CONCEPTS EXPLAINED IN THIS SCRIPT
=============================================================================

1. Process Group Initialization (init_process_group)
   - Creates communication backend (NCCL for GPU, Gloo for CPU)
   - Every process must call this to join the distributed group

2. Ranks and World Size
   - rank: Unique ID for this process (0 to world_size-1)
   - local_rank: GPU index on this machine (0 to ngpus_per_node-1)
   - world_size: Total number of processes across all nodes

3. Model Sharding with fully_shard()
   - Wraps model to shard parameters across GPUs
   - Can be applied per-layer (fine-grained) or whole-model (coarse)
   - Parameters are DTensors (distributed tensors)

4. Forward/Backward Pass
   - Forward: Each GPU all-gathers needed params, computes, then frees
   - Backward: Same gather pattern, computes gradients, shards them
   - Communication happens automatically!

5. Optimizer State Sharding
   - Each GPU only stores optimizer state for its shard
   - AdamW: momentum + variance for 1/N of parameters
   - Massive memory savings for large models

6. Checkpointing
   - Must gather full state dict on rank 0 for saving
   - Or use distributed checkpointing (more advanced)

=============================================================================
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed._tensor import distribute_tensor
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

# Import your existing model and data utilities
# If these don't exist, we'll define simplified versions below
try:
    from gpt.model import GPT
    from gpt.config import GPTConfig
    from gpt.data import DataLoaderLite, LoaderConfig
except ImportError:
    # Fallback: define minimal versions for demonstration
    print("Warning: Could not import GPT model/data. Using placeholder.")
    GPT = None
    GPTConfig = None
    DataLoaderLite = None
    LoaderConfig = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FSDPTrainingConfig:
    """
    Configuration for FSDP training.

    This dataclass holds all hyperparameters and settings for distributed training.
    Using a dataclass makes it easy to:
    - See all settings in one place
    - Override via command-line arguments
    - Pass config around cleanly
    """
    # Data
    data_root: str = "data/fineweb10B"

    # Model architecture (GPT-2 124M by default)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    block_size: int = 1024

    # Training hyperparameters
    micro_batch_size: int = 8      # Batch size per GPU
    seq_len: int = 1024             # Sequence length
    grad_accum_steps: int = 4       # Gradient accumulation steps
    max_steps: int = 10000          # Total training steps

    # Optimizer
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Learning rate schedule
    warmup_steps: int = 100

    # FSDP-specific settings
    sharding_strategy: str = "full"  # "full" or "hybrid" or "grad_op"
    cpu_offload: bool = False        # Offload params to CPU (slower, saves GPU memory)

    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 20
    save_interval: int = 1000
    output_dir: str = "log_fsdp"

    # Mixed precision
    use_amp: bool = True             # Use mixed precision training


# =============================================================================
# Distributed Utilities
# =============================================================================

def setup_distributed():
    """
    Initialize the distributed process group.

    This function MUST be called by every process before any distributed operations.

    Process group: A collection of processes that can communicate with each other.
    - Backend: Communication library (NCCL for GPU, Gloo for CPU)
    - NCCL: NVIDIA Collective Communications Library - highly optimized for GPU

    Environment variables set by torchrun:
    - RANK: Global rank of this process (0 to world_size-1)
    - LOCAL_RANK: Rank on this node (0 to ngpus_per_node-1)
    - WORLD_SIZE: Total number of processes
    - MASTER_ADDR: IP address of rank 0 process
    - MASTER_PORT: Port for communication

    Returns:
        Tuple of (rank, local_rank, world_size, is_master)
    """
    # Check if we're in distributed mode
    # torchrun sets RANK and WORLD_SIZE environment variables
    if 'RANK' not in os.environ:
        # Not launched with torchrun - single GPU mode
        print("Warning: Not launched with torchrun. Running in single-GPU mode.")
        return 0, 0, 1, True

    # Get process ranks from environment
    rank = int(os.environ['RANK'])              # Global rank across all nodes
    local_rank = int(os.environ['LOCAL_RANK'])  # Local rank on this node (which GPU)
    world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes

    # Initialize process group
    # This creates the communication backend and connects all processes
    # All processes must call this before any distributed operations
    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPU communication (fastest)
        init_method='env://',  # Use environment variables for coordination
    )

    # Set the CUDA device for this process
    # Each process should use a different GPU (local_rank = GPU index)
    # This is CRITICAL - without this, all processes would use GPU 0!
    torch.cuda.set_device(local_rank)

    # Only rank 0 should print logs (avoid duplicate output)
    is_master = rank == 0

    if is_master:
        print(f"==> Initialized distributed training:")
        print(f"    World size: {world_size}")
        print(f"    Rank: {rank}")
        print(f"    Local rank: {local_rank}")
        print(f"    Device: cuda:{local_rank}")

    return rank, local_rank, world_size, is_master


def cleanup_distributed():
    """
    Clean up the distributed process group.

    This should be called at the end of training to properly shut down
    the communication backend and free resources.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device_mesh(world_size: int) -> DeviceMesh:
    """
    Create a DeviceMesh for FSDP2.

    DeviceMesh is a new concept in PyTorch 2.x that represents the topology
    of devices (GPUs) in a structured way. It's used by FSDP2 to determine
    how to shard and communicate tensors.

    For single-node training, we create a 1D mesh with all GPUs in a row:
        [GPU0, GPU1, GPU2, GPU3]

    For multi-node training, you could create a 2D mesh:
        [[GPU0, GPU1],   # Node 0
         [GPU2, GPU3]]   # Node 1

    Args:
        world_size: Total number of GPUs

    Returns:
        DeviceMesh object representing device topology
    """
    # Create a 1D mesh with all GPUs (simple single-node setup)
    # The mesh is just a list of device ranks: [0, 1, 2, ..., world_size-1]
    return DeviceMesh("cuda", list(range(world_size)))


# =============================================================================
# Model Setup with FSDP2
# =============================================================================

def apply_fsdp(model: nn.Module, config: FSDPTrainingConfig) -> nn.Module:
    """
    Apply FSDP2 sharding to the model using fully_shard().

    This is the HEART of FSDP2! Here's what happens when you call fully_shard():

    BEFORE fully_shard():
    - model.parameters() are regular torch.Tensor on a single GPU
    - Each parameter occupies GPU memory
    - Example: A 768x768 weight matrix = 768*768*4 bytes = 2.3 MB

    AFTER fully_shard():
    - model.parameters() become DTensor (Distributed Tensor)
    - Each parameter is SHARDED across all GPUs in the mesh
    - Example: Same 768x768 matrix on 4 GPUs:
        GPU 0: [192x768] shard (1/4 of the rows)
        GPU 1: [192x768] shard
        GPU 2: [192x768] shard
        GPU 3: [192x768] shard
    - Each GPU stores only 192*768*4 bytes = 0.58 MB (4x memory savings!)

    During forward/backward:
    - When a layer is used, FSDP2 all-gathers the full parameter temporarily
    - After computation, the gathered parameter is freed
    - Only the shard is kept in memory

    This means:
    - Minimal memory usage when not computing (only shards)
    - Full parameter available when needed (all-gather)
    - Automatic gradient sharding during backward pass

    Sharding strategies:
    - "full": Shard params, gradients, AND optimizer states (maximum memory savings)
    - "grad_op": Only shard gradients and optimizer states (params replicated)
    - "hybrid": Combination for multi-node setups

    Args:
        model: The model to shard
        config: Training configuration

    Returns:
        FSDP-wrapped model with sharded parameters
    """
    # Set up mixed precision policy
    # This tells FSDP what dtypes to use for params, gradients, and computation
    if config.use_amp:
        # Mixed precision setup:
        # - param_dtype: How to store parameters (bfloat16 saves memory)
        # - reduce_dtype: What dtype to use for gradient reduction (float32 for stability)
        # - buffer_dtype: What dtype for buffers like LayerNorm stats (float32 for accuracy)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,      # Store params in bfloat16 (half precision)
            reduce_dtype=torch.float32,      # Reduce gradients in fp32 (stability)
        )
    else:
        # No mixed precision - use fp32 everywhere
        mp_policy = None

    # OPTION 1: Whole-model sharding (simplest, coarse-grained)
    # This wraps the entire model in one FSDP unit
    # Good for: Smaller models, simplicity
    #
    # model = fully_shard(model, mixed_precision=mp_policy)

    # OPTION 2: Per-layer sharding (recommended, fine-grained)
    # This wraps each transformer block separately, then wraps the whole model
    # Good for: Better communication overlap, more memory efficiency
    #
    # For GPT-2, we have:
    # - model.transformer.wte (embedding)
    # - model.transformer.wpe (positional embedding)
    # - model.transformer.h[0..11] (transformer blocks) <- shard these individually!
    # - model.lm_head (output layer)

    print("Applying FSDP2 with fully_shard()...")

    # First, shard each transformer block individually
    # This creates fine-grained FSDP units for better performance
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        for i, block in enumerate(model.transformer.h):
            # Each block becomes its own FSDP unit
            # During forward pass, only one block's params are gathered at a time
            # This reduces peak memory usage significantly!
            model.transformer.h[i] = fully_shard(
                block,
                mixed_precision=mp_policy,
            )
            if i == 0:
                print(f"  ✓ Sharded transformer block {i} (showing only first)")
        print(f"  ✓ Sharded {len(model.transformer.h)} transformer blocks")

    # Then, wrap the entire model
    # This creates a root FSDP unit that manages the embeddings and output layer
    model = fully_shard(
        model,
        mixed_precision=mp_policy,
    )
    print("  ✓ Sharded root model")

    print("\n" + "="*70)
    print("FSDP2 SHARDING COMPLETE")
    print("="*70)
    print("What just happened:")
    print("1. Each transformer block is now an independent FSDP unit")
    print("2. Parameters are sharded across all GPUs using DTensor")
    print("3. During forward pass:")
    print("   - Block 0's params are all-gathered → compute → freed")
    print("   - Block 1's params are all-gathered → compute → freed")
    print("   - ... (sequential gather-compute-free)")
    print("4. During backward pass:")
    print("   - Same pattern in reverse")
    print("   - Gradients are automatically sharded")
    print("5. Optimizer states are also sharded (huge memory savings!)")
    print("="*70 + "\n")

    return model


# =============================================================================
# Learning Rate Schedule
# =============================================================================

def get_lr(step: int, config: FSDPTrainingConfig) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    This is identical to non-distributed training - FSDP doesn't change
    the learning rate schedule at all!

    Args:
        step: Current training step
        config: Training configuration

    Returns:
        Learning rate for this step
    """
    # Linear warmup
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    # Cosine decay after warmup
    if step > config.max_steps:
        return config.learning_rate * 0.1

    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# Checkpointing with FSDP
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: FSDPTrainingConfig,
    rank: int,
):
    """
    Save model and optimizer checkpoint.

    FSDP CHECKPOINTING CHALLENGE:
    - Parameters are sharded across GPUs as DTensors
    - Optimizer states are also sharded
    - We need to gather everything to rank 0 for saving

    FSDP2 provides utilities for this, but for simplicity, we'll show
    the basic approach: only rank 0 saves, and we gather the state dict.

    For production, consider:
    - torch.distributed.checkpoint (distributed checkpointing)
    - Saves sharded checkpoints (faster, more scalable)
    - But requires more complex loading logic

    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer
        step: Current step
        config: Training config
        rank: Process rank
    """
    if rank != 0:
        # Only rank 0 saves checkpoints
        return

    os.makedirs(config.output_dir, exist_ok=True)

    # For FSDP2, we need to gather the full state dict to rank 0
    # There are several options:
    # 1. model.state_dict() with special FSDP context (recommended)
    # 2. Manually gather DTensors (complex)
    # 3. Use torch.distributed.checkpoint (advanced)

    # For now, we'll use the simplest approach:
    # FSDP2's state_dict() automatically handles gathering

    print(f"Saving checkpoint at step {step}...")

    checkpoint = {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
    }

    checkpoint_path = os.path.join(config.output_dir, f"checkpoint_{step:06d}.pt")
    torch.save(checkpoint, checkpoint_path)

    print(f"  ✓ Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    rank: int,
) -> int:
    """
    Load model and optimizer checkpoint.

    FSDP LOADING:
    - All ranks must participate in loading (not just rank 0)
    - Each rank loads its shard of the parameters
    - FSDP2 handles the distribution automatically

    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint
        rank: Process rank

    Returns:
        Starting step number
    """
    if not os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        return 0

    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}...")

    # Load checkpoint on CPU first to avoid OOM
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model state dict
    # FSDP2 will automatically shard the loaded parameters
    model.load_state_dict(checkpoint['model'])

    # Load optimizer state dict
    optimizer.load_state_dict(checkpoint['optimizer'])

    step = checkpoint['step']

    if rank == 0:
        print(f"  ✓ Loaded checkpoint from step {step}")

    return step


# =============================================================================
# Training Loop
# =============================================================================

def train(config: FSDPTrainingConfig):
    """
    Main training function with FSDP2.

    This is the core training loop that demonstrates how FSDP2 works
    in practice. The key insight: FSDP2 is mostly transparent! Once
    you've wrapped the model with fully_shard(), training looks almost
    identical to single-GPU training.

    The main differences:
    1. Setup: init_process_group, set device
    2. Model: wrap with fully_shard()
    3. Data: each rank gets different data
    4. Logging: only rank 0 logs
    5. Checkpointing: gather state dict to rank 0
    6. Cleanup: destroy_process_group at end

    Args:
        config: Training configuration
    """
    # -------------------------------------------------------------------------
    # 1. DISTRIBUTED SETUP
    # -------------------------------------------------------------------------

    rank, local_rank, world_size, is_master = setup_distributed()
    device = f'cuda:{local_rank}'

    if is_master:
        print("\n" + "="*70)
        print("FSDP2 TRAINING START")
        print("="*70)
        print(f"World size: {world_size} GPUs")
        print(f"Batch size per GPU: {config.micro_batch_size}")
        print(f"Gradient accumulation: {config.grad_accum_steps}")
        print(f"Effective batch size: {config.micro_batch_size * world_size * config.grad_accum_steps}")
        print("="*70 + "\n")

    # -------------------------------------------------------------------------
    # 2. CREATE MODEL
    # -------------------------------------------------------------------------

    if is_master:
        print("Creating model...")

    # Create model on the correct device
    # For FSDP2, we can create on CUDA directly (meta device is optional)
    if GPT is not None and GPTConfig is not None:
        model_config = GPTConfig(
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            vocab_size=config.vocab_size,
            block_size=config.block_size,
        )
        model = GPT(model_config)
    else:
        # Fallback: create a dummy model for demonstration
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(768, 768) for _ in range(12)
                ])
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x, None
        model = DummyModel()

    # Move to device BEFORE applying FSDP
    # This is important: create model on correct device first
    model = model.to(device)

    if is_master:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params/1e6:.2f}M")
        print(f"  Model memory (before FSDP): {total_params*4/1024**2:.2f} MB per GPU")
        print(f"  Model memory (after FSDP): ~{total_params*4/world_size/1024**2:.2f} MB per GPU (sharded)")

    # -------------------------------------------------------------------------
    # 3. APPLY FSDP2 SHARDING
    # -------------------------------------------------------------------------

    # THIS IS THE KEY STEP!
    # fully_shard() transforms the model from regular parameters to DTensors
    model = apply_fsdp(model, config)

    # -------------------------------------------------------------------------
    # 4. CREATE OPTIMIZER
    # -------------------------------------------------------------------------

    if is_master:
        print("\nCreating optimizer...")

    # Create optimizer AFTER applying FSDP
    # The optimizer will automatically work with sharded parameters!
    # Each GPU only optimizes its shard of parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    if is_master:
        print(f"  Optimizer: AdamW")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Optimizer state memory: ~{total_params*8/world_size/1024**2:.2f} MB per GPU (sharded)")
        print(f"    (Each GPU stores momentum + variance for 1/{world_size} of params)")

    # -------------------------------------------------------------------------
    # 5. CREATE DATA LOADERS
    # -------------------------------------------------------------------------

    if is_master:
        print("\nCreating data loaders...")

    # Each rank gets a different portion of the data
    # This is automatic with process_rank and world_size
    if DataLoaderLite is not None and LoaderConfig is not None:
        train_loader = DataLoaderLite(LoaderConfig(
            batch_size=config.micro_batch_size,
            seq_len=config.seq_len,
            process_rank=rank,
            world_size=world_size,
            split="train",
            data_root=config.data_root,
            master_process=is_master,
        ))

        val_loader = DataLoaderLite(LoaderConfig(
            batch_size=config.micro_batch_size,
            seq_len=config.seq_len,
            process_rank=rank,
            world_size=world_size,
            split="val",
            data_root=config.data_root,
            master_process=is_master,
        ))
    else:
        # Fallback: dummy data
        if is_master:
            print("  Warning: Using dummy data (no DataLoaderLite found)")
        train_loader = None
        val_loader = None

    # -------------------------------------------------------------------------
    # 6. TRAINING LOOP
    # -------------------------------------------------------------------------

    if is_master:
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70 + "\n")

    model.train()
    step = 0

    while step < config.max_steps:
        t0 = time.time()

        # Zero gradients
        optimizer.zero_grad()

        # Gradient accumulation loop
        loss_accum = 0.0
        for micro_step in range(config.grad_accum_steps):
            # Get batch
            if train_loader is not None:
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
            else:
                # Dummy data
                x = torch.randint(0, config.vocab_size, (config.micro_batch_size, config.seq_len), device=device)
                y = torch.randint(0, config.vocab_size, (config.micro_batch_size, config.seq_len), device=device)

            # Forward pass
            # FSDP2 automatically:
            # 1. All-gathers parameters for each layer as needed
            # 2. Computes forward pass
            # 3. Frees gathered parameters after computation
            # 4. Keeps only the shard in memory

            if config.use_amp:
                # Mixed precision forward pass
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)

            # Scale loss for gradient accumulation
            loss = loss / config.grad_accum_steps
            loss_accum += loss.detach()

            # Backward pass
            # FSDP2 automatically:
            # 1. All-gathers parameters again (in reverse order)
            # 2. Computes gradients
            # 3. Shards gradients across GPUs (reduce-scatter)
            # 4. Each GPU keeps only its gradient shard
            loss.backward()

        # All-reduce the loss across ranks for logging
        # This ensures all ranks see the same loss value
        if world_size > 1:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Gradient clipping
        # Note: For FSDP, we clip the norm of the SHARDED gradients
        # This is correct - each rank clips its own shard
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Update learning rate
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Optimizer step
        # Each GPU updates only its shard of parameters!
        # No synchronization needed - each shard is independent
        optimizer.step()

        # Timing
        t1 = time.time()
        dt = (t1 - t0) * 1000  # milliseconds

        # Tokens per second across all GPUs
        tokens_per_batch = config.micro_batch_size * config.seq_len * config.grad_accum_steps * world_size
        tokens_per_sec = tokens_per_batch / (t1 - t0)

        # Logging (only rank 0)
        if is_master and step % config.log_interval == 0:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.2e} | "
                  f"dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.0f}")

        # Validation (only rank 0 for logging, but all ranks participate)
        if step % config.eval_interval == 0 and step > 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for _ in range(config.eval_iters):
                    if val_loader is not None:
                        x, y = val_loader.next_batch()
                        x, y = x.to(device), y.to(device)
                    else:
                        x = torch.randint(0, config.vocab_size, (config.micro_batch_size, config.seq_len), device=device)
                        y = torch.randint(0, config.vocab_size, (config.micro_batch_size, config.seq_len), device=device)

                    if config.use_amp:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            _, loss = model(x, y)
                    else:
                        _, loss = model(x, y)

                    val_loss += loss.detach()

            val_loss = val_loss / config.eval_iters

            # All-reduce validation loss
            if world_size > 1:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            if is_master:
                print(f"{'='*70}")
                print(f"step {step} | val_loss: {val_loss.item():.6f}")
                print(f"{'='*70}")

            model.train()

        # Checkpointing
        if step % config.save_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, config, rank)

        step += 1

    # -------------------------------------------------------------------------
    # 7. CLEANUP
    # -------------------------------------------------------------------------

    if is_master:
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)

    # Save final checkpoint
    save_checkpoint(model, optimizer, step, config, rank)

    # Clean up distributed
    cleanup_distributed()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Parse arguments and launch training.

    This is the entry point when you run:
        torchrun --nproc_per_node=4 train_fsdp.py
    """
    parser = argparse.ArgumentParser(description="Train GPT-2 with FSDP2")

    # Data
    parser.add_argument("--data-root", type=str, default="data/fineweb10B",
                        help="Path to training data")

    # Model
    parser.add_argument("--n-layer", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, default=768,
                        help="Embedding dimension")

    # Training
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=6e-4,
                        help="Learning rate")

    # FSDP
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload parameters to CPU (saves GPU memory, slower)")

    # Logging
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--eval-interval", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--output-dir", type=str, default="log_fsdp",
                        help="Output directory for checkpoints")

    args = parser.parse_args()

    # Create config from arguments
    config = FSDPTrainingConfig(
        data_root=args.data_root,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        micro_batch_size=args.batch_size,
        seq_len=args.seq_len,
        grad_accum_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        use_amp=not args.no_amp,
        cpu_offload=args.cpu_offload,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
    )

    # Launch training
    train(config)


if __name__ == "__main__":
    main()
