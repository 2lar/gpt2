"""Distributed Training Utilities

Handles setup and teardown of PyTorch Distributed Data Parallel (DDP) training.
Supports modern torch.accelerator API (PyTorch 2.8+) with fallback to torch.cuda.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


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
