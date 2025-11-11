"""Training Configuration

Shared configuration dataclass for training scripts.
Both onefile_train.py and train.py use this to ensure consistency.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration.

    This configuration is shared between both training versions
    (onefile_train.py and modular train.py) to ensure consistency.
    """

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
