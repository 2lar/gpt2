"""Data Loading Utilities

Helper functions for creating and configuring data loaders.
"""
from __future__ import annotations

from typing import Tuple

from gpt.data import DataLoaderLite, LoaderConfig


def build_dataloaders(
    batch_size: int,
    seq_len: int,
    data_root: str,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    master_process: bool = True
) -> Tuple[DataLoaderLite, DataLoaderLite]:
    """
    Build training and validation data loaders.

    Args:
        batch_size: Micro batch size
        seq_len: Sequence length
        data_root: Root directory containing data shards
        ddp_rank: DDP process rank
        ddp_world_size: Total number of DDP processes
        master_process: Whether this is the master process

    Returns:
        (train_loader, val_loader) tuple
    """
    common = dict(
        batch_size=batch_size,
        seq_len=seq_len,
        process_rank=ddp_rank,
        world_size=ddp_world_size,
        data_root=data_root,
        master_process=master_process,
    )

    train_loader = DataLoaderLite(LoaderConfig(split="train", **common))
    val_loader = DataLoaderLite(LoaderConfig(split="val", **common))

    return train_loader, val_loader
