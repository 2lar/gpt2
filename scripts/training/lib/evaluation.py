"""Evaluation Utilities

Functions for evaluating model performance during training:
- Validation loss computation
- HellaSwag benchmark evaluation
- Text generation sampling
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Callable

import torch
import torch.nn as nn
import torch.distributed as dist

if TYPE_CHECKING:
    from gpt.data import DataLoaderLite


def evaluate_loss(
    model: nn.Module,
    val_loader: 'DataLoaderLite',
    eval_iters: int,
    device: str,
    device_type: str = 'cuda'
) -> float:
    """
    Evaluate validation loss.

    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        eval_iters: Number of validation batches to average over
        device: Device to run evaluation on
        device_type: Device type for autocast ('cuda' or 'cpu')

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)

            total_loss += loss.item()

    model.train()
    return total_loss / eval_iters


def run_hellaswag_eval(
    model: nn.Module,
    iterate_examples: Callable,
    render_example: Callable,
    get_most_likely_row: Callable,
    device: str,
    device_type: str,
    ddp_rank: int = 0,
    ddp_world_size: int = 1
) -> tuple[int, int]:
    """
    Run HellaSwag benchmark evaluation.

    Args:
        model: The model to evaluate
        iterate_examples: Function to iterate through examples
        render_example: Function to render example as tokens
        get_most_likely_row: Function to get most likely answer
        device: Device to run evaluation on
        device_type: Device type for autocast
        ddp_rank: DDP rank (for distributed evaluation)
        ddp_world_size: DDP world size

    Returns:
        (num_correct, num_total) tuple
    """
    num_correct_norm = 0
    num_total = 0

    for i, example in enumerate(iterate_examples("val")):
        # Distribute work across GPUs
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

    return num_correct_norm, num_total


def generate_samples(
    model: nn.Module,
    tokenizer,
    prompt: str,
    num_samples: int,
    max_length: int,
    device: str,
    device_type: str,
    ddp_rank: int = 0
) -> list[str]:
    """
    Generate text samples from the model.

    Args:
        model: The model to use for generation
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt to start generation
        num_samples: Number of samples to generate
        max_length: Maximum length of generated text
        device: Device to run generation on
        device_type: Device type for autocast
        ddp_rank: DDP rank (for seeding)

    Returns:
        List of generated text samples
    """
    model.eval()

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    tokens = tokens.repeat(num_samples, 1)
    xgen = tokens.to(device)

    # Setup RNG for reproducibility
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)

    # Generate tokens
    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(xgen)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)

    # Decode samples
    samples = []
    for i in range(num_samples):
        tokens_list = xgen[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens_list)
        samples.append(decoded)

    model.train()
    return samples
