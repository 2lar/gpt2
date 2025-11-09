"""
evaluation.py: Helper utilities for validation scoring (HellaSwag etc.).
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def get_most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int:
    """Mirror the lecture helper that scores HellaSwag completions."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    return avg_loss.argmin().item()
