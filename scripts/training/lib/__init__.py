"""Training Library - Modular Training Utilities

This package contains reusable training utilities extracted from the monolithic
training script for better separation of concerns.
"""

from .config import TrainingConfig
from .distributed import setup_distributed, teardown_distributed
from .scheduling import get_lr
from .evaluation import evaluate_loss, run_hellaswag_eval, generate_samples
from .data import build_dataloaders

__all__ = [
    'TrainingConfig',
    'setup_distributed',
    'teardown_distributed',
    'get_lr',
    'evaluate_loss',
    'run_hellaswag_eval',
    'generate_samples',
    'build_dataloaders',
]
