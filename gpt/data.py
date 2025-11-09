"""
Data Loading Utilities

Provides efficient data loading for language model training using pre-tokenized
shards. Supports distributed training with multiple processes.

The data is expected to be stored as .npy files containing tokenized text,
with separate shards for training and validation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import torch

TensorPair = Tuple[torch.Tensor, torch.Tensor]


def load_tokens(filename: str) -> torch.Tensor:
    """
    Load a shard of pre-tokenized data from disk.

    Tokens are stored as NumPy arrays (.npy files) for efficient storage
    and fast loading. Each shard contains millions of tokens.

    Args:
        filename: Path to .npy file containing token IDs

    Returns:
        PyTorch tensor of token IDs (dtype=long for indexing)
    """
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # Ensure consistent dtype
    return torch.tensor(npt, dtype=torch.long)


@dataclass
class LoaderConfig:
    batch_size: int
    seq_len: int
    process_rank: int
    world_size: int
    split: str
    data_root: str = "shakespeare_data"
    master_process: bool = True


class DataLoaderLite:
    """
    Lightweight data loader for language model training.

    Loads pre-tokenized data from shards (large .npy files) and yields batches
    for training. Supports distributed training by dividing data across processes.

    Key features:
    - Automatic shard cycling (loops through all shards indefinitely)
    - Distributed-friendly (each process reads different portions)
    - Memory efficient (loads one shard at a time)
    """

    def __init__(self, cfg: LoaderConfig):
        assert cfg.split in {"train", "val"}, "split must be 'train' or 'val'"
        self.cfg = cfg

        # Find all shard files for this split
        # Files are expected to be named like: "train_00001.npy", "val_00001.npy"
        shards = sorted(
            os.path.join(cfg.data_root, s)
            for s in os.listdir(cfg.data_root)
            if cfg.split in s
        )
        if len(shards) == 0:
            raise FileNotFoundError(f"no shards found for split {cfg.split} in {cfg.data_root}")
        if cfg.master_process:
            print(f"found {len(shards)} shards for split {cfg.split}")
        self.shards = shards
        self.reset()

    @property
    def B(self) -> int:  # Batch size (match reference naming convention)
        return self.cfg.batch_size

    @property
    def T(self) -> int:  # Sequence length
        return self.cfg.seq_len

    def reset(self) -> None:
        """Reset to the beginning of the first shard."""
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])

        # In distributed training, each process starts at a different offset
        # This ensures processes don't duplicate data
        self.current_position = self.B * self.T * self.cfg.process_rank

    def _advance_shard(self) -> None:
        """Move to the next shard, wrapping around if at the end."""
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.cfg.process_rank

    def next_batch(self) -> TensorPair:
        """
        Get the next batch of training data.

        Returns (inputs, targets) where targets are inputs shifted by 1 position.
        This is the standard setup for language modeling: predict the next token.

        Returns:
            x: Input tokens, shape (B, T)
            y: Target tokens (x shifted by 1), shape (B, T)
        """
        B, T = self.B, self.T
        start = self.current_position
        stop = start + B * T + 1  # +1 to get targets

        # Extract a chunk of tokens: [start, start+1, ..., start+B*T]
        buf = self.tokens[start:stop]

        # Create input-target pairs by shifting
        # x = [0, 1, 2, ..., B*T-1], y = [1, 2, 3, ..., B*T]
        # Each input token predicts the next token
        x = buf[:-1].view(B, T)  # Drop last token, reshape to (B, T)
        y = buf[1:].view(B, T)   # Drop first token, reshape to (B, T)

        # Advance position by (B * T * world_size)
        # In distributed training, this skips over the data used by other processes
        self.current_position += B * T * self.cfg.world_size

        # Check if we'll run out of tokens in the next batch
        # If so, move to the next shard
        if self.current_position + (B * T * self.cfg.world_size + 1) > len(self.tokens):
            self._advance_shard()

        return x, y

    def __iter__(self) -> Iterator[TensorPair]:
        """Infinite iterator over batches (for training)."""
        while True:
            yield self.next_batch()
