"""
### JUST TO NOTE, I'm doing this instead of using hugging face library for DPO which
is like 10 lines of code insteasd, this is more about understanding under the hood.


train_dpo.py: Train GPT-2 with DPO (Direct Preference Optimization) from scratch

DPO is a method for aligning language models to human preferences WITHOUT
reinforcement learning. Instead of RLHF (which requires a reward model + PPO),
DPO trains directly on preference pairs.

=============================================================================
WHAT IS DPO?
=============================================================================

Given preference data like:
  Prompt: "Explain quantum computing"
  Chosen: "Quantum computing uses qubits in superposition..." ✓ (better)
  Rejected: "Quantum is like fast computers." ✗ (worse)

DPO trains the model to:
1. Increase probability of "chosen" responses
2. Decrease probability of "rejected" responses
3. Stay close to the original model (prevent drift)

No reward model needed! No RL needed! Just standard supervised learning.

=============================================================================
THE DPO ALGORITHM
=============================================================================

1. Start with a pretrained model (e.g., GPT-2)
2. Create a frozen reference copy (prevents drift)
3. For each preference pair (prompt, chosen, rejected):

   a) Compute log P(chosen | prompt) under current model
   b) Compute log P(rejected | prompt) under current model
   c) Compute same for reference model

   d) Calculate implicit reward:
      reward_chosen = log π_θ(chosen) - log π_ref(chosen)
      reward_rejected = log π_θ(rejected) - log π_ref(rejected)

   e) DPO loss:
      loss = -log(sigmoid(β * (reward_chosen - reward_rejected)))

   Where β controls how strongly to prefer chosen over rejected

4. Backpropagate and update model (NOT reference!)
5. Repeat until converged

Result: Model that prefers "chosen" responses while staying close to reference.

=============================================================================
USAGE
=============================================================================

Basic usage:
    python -m scripts.training.train_dpo \
      --model gpt2 \
      --data preference_data.jsonl \
      --steps 1000

With custom settings:
    python -m scripts.training.train_dpo \
      --model gpt2 \
      --weights data/gpt2_weights.pt \
      --data preference_data.jsonl \
      --beta 0.1 \
      --lr 5e-7 \
      --steps 1000 \
      --output dpo_checkpoints

Expected data format (JSONL):
    {"prompt": "Explain X", "chosen": "Good response", "rejected": "Bad response"}
    {"prompt": "What is Y?", "chosen": "Detailed answer", "rejected": "Short answer"}
    ...

=============================================================================
KEY CONCEPTS EXPLAINED
=============================================================================

1. Reference Model (π_ref)
   - Frozen copy of the model at the start of DPO training
   - Acts as a "regularizer" to prevent the model from drifting too far
   - Essential: without it, model could "cheat" by becoming very confident

2. Beta (β) Parameter
   - Controls the strength of preference learning
   - Higher β → stronger preference for chosen over rejected
   - Lower β → more conservative (stays closer to reference)
   - Typical values: 0.1 to 0.5

3. Log Probabilities
   - We work with log probabilities (not raw probabilities) for numerical stability
   - log P(sequence) = sum of log P(each token)
   - Easier to compute and more stable

4. Implicit Reward
   - DPO implicitly defines a reward function
   - r(x, y) = β * (log π_θ(y|x) - log π_ref(y|x))
   - Higher reward = model assigns higher probability than reference

5. Why DPO Works
   - Maximizes likelihood of chosen responses
   - Minimizes likelihood of rejected responses
   - KL penalty (implicit) keeps model close to reference
   - All in one simple loss function!

=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from gpt.model import GPT
from gpt.config import GPTConfig


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    # Model
    model_type: str = "gpt2"
    weights_path: str = "data/gpt2_weights.pt"

    # Data
    data_path: str = "preference_data.jsonl"
    val_split: float = 0.1  # Fraction of data for validation

    # DPO-specific hyperparameters
    beta: float = 0.1  # Temperature parameter (higher = stronger preference)

    # Training
    batch_size: int = 4  # Small batch for limited data
    learning_rate: float = 5e-7  # Very low LR for stability (model already pretrained)
    max_steps: int = 1000
    grad_accum_steps: int = 4
    warmup_steps: int = 50

    # Sequence lengths
    max_prompt_length: int = 512
    max_length: int = 1024  # Max total length (prompt + response)

    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: str = "dpo_checkpoints"


# =============================================================================
# Preference Dataset
# =============================================================================

class PreferenceDataset(Dataset):
    """
    Dataset for preference pairs.

    Each example contains:
    - prompt: The input prompt
    - chosen: The preferred response
    - rejected: The less preferred response

    File format (JSONL):
        {"prompt": "...", "chosen": "...", "rejected": "..."}
        {"prompt": "...", "chosen": "...", "rejected": "..."}
        ...
    """

    def __init__(self, data_path: str, tokenizer, max_prompt_length: int, max_length: int):
        """
        Load preference data from JSONL file.

        Args:
            data_path: Path to JSONL file with preference pairs
            tokenizer: Tokenizer for encoding text
            max_prompt_length: Maximum prompt length
            max_length: Maximum total length (prompt + response)
        """
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_length = max_length

        # Load data
        self.examples = []

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Preference data not found at {data_path}\n"
                f"Expected JSONL format with keys: 'prompt', 'chosen', 'rejected'\n"
                f"Example: {{'prompt': 'Explain X', 'chosen': 'Good answer', 'rejected': 'Bad answer'}}"
            )

        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    example = json.loads(line)

                    # Validate keys
                    required_keys = ['prompt', 'chosen', 'rejected']
                    if not all(k in example for k in required_keys):
                        raise ValueError(f"Missing required keys. Expected: {required_keys}, Got: {list(example.keys())}")

                    self.examples.append(example)

        print(f"Loaded {len(self.examples)} preference pairs from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single preference pair.

        Returns a dict with:
        - prompt_ids: Tokenized prompt
        - chosen_ids: Tokenized chosen response
        - rejected_ids: Tokenized rejected response
        - chosen_labels: Labels for chosen (prompt masked as -100)
        - rejected_labels: Labels for rejected (prompt masked as -100)
        """
        example = self.examples[idx]

        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(example['prompt'])
        prompt_tokens = prompt_tokens[:self.max_prompt_length]  # Truncate if too long

        # Tokenize chosen response (full = prompt + chosen)
        chosen_tokens = self.tokenizer.encode(example['chosen'])
        max_chosen_len = self.max_length - len(prompt_tokens)
        chosen_tokens = chosen_tokens[:max_chosen_len]

        # Tokenize rejected response (full = prompt + rejected)
        rejected_tokens = self.tokenizer.encode(example['rejected'])
        max_rejected_len = self.max_length - len(prompt_tokens)
        rejected_tokens = rejected_tokens[:max_rejected_len]

        # Concatenate prompt + response
        chosen_full = prompt_tokens + chosen_tokens
        rejected_full = prompt_tokens + rejected_tokens

        # Create labels (mask prompt tokens with -100, only compute loss on response)
        # This is CRITICAL: we only want to compute loss on the response, not the prompt
        chosen_labels = [-100] * len(prompt_tokens) + chosen_tokens
        rejected_labels = [-100] * len(prompt_tokens) + rejected_tokens

        return {
            'chosen_input_ids': torch.tensor(chosen_full, dtype=torch.long),
            'chosen_labels': torch.tensor(chosen_labels, dtype=torch.long),
            'rejected_input_ids': torch.tensor(rejected_full, dtype=torch.long),
            'rejected_labels': torch.tensor(rejected_labels, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Pads sequences to the same length within the batch.
    """
    # Find max lengths in this batch
    max_chosen_len = max(ex['chosen_input_ids'].size(0) for ex in batch)
    max_rejected_len = max(ex['rejected_input_ids'].size(0) for ex in batch)

    # Pad sequences
    chosen_input_ids = []
    chosen_labels = []
    rejected_input_ids = []
    rejected_labels = []

    for ex in batch:
        # Pad chosen
        chosen_pad_len = max_chosen_len - ex['chosen_input_ids'].size(0)
        chosen_input_ids.append(F.pad(ex['chosen_input_ids'], (0, chosen_pad_len), value=0))
        chosen_labels.append(F.pad(ex['chosen_labels'], (0, chosen_pad_len), value=-100))

        # Pad rejected
        rejected_pad_len = max_rejected_len - ex['rejected_input_ids'].size(0)
        rejected_input_ids.append(F.pad(ex['rejected_input_ids'], (0, rejected_pad_len), value=0))
        rejected_labels.append(F.pad(ex['rejected_labels'], (0, rejected_pad_len), value=-100))

    return {
        'chosen_input_ids': torch.stack(chosen_input_ids),
        'chosen_labels': torch.stack(chosen_labels),
        'rejected_input_ids': torch.stack(rejected_input_ids),
        'rejected_labels': torch.stack(rejected_labels),
    }


# =============================================================================
# DPO Loss Function
# =============================================================================

def compute_sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probability of a sequence under the model.

    This is the core computation for DPO: we need to know how likely the model
    thinks a particular sequence is.

    Args:
        model: Language model
        input_ids: Input token IDs [batch, seq_len]
        labels: Target token IDs [batch, seq_len]
                Use -100 for tokens that should be ignored (e.g., prompt tokens)

    Returns:
        Log probability of each sequence in the batch [batch]

    Example:
        For sequence "The cat sat on the mat":
        - log P(sequence) = log P(cat|The) + log P(sat|The,cat) + ...
        - We sum log probabilities of each token given previous tokens
    """
    # Forward pass through model
    # Note: GPT models shift labels internally, so we pass input_ids directly
    outputs = model(input_ids)
    logits = outputs[0]  # [batch, seq_len, vocab_size]

    # Shift logits and labels for next-token prediction
    # logits: predict token at position i
    # labels: ground truth is token at position i+1
    shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
    shift_labels = labels[..., 1:].contiguous()      # [batch, seq_len-1]

    # Compute log probabilities
    # We use log_softmax for numerical stability
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [batch, seq_len-1, vocab_size]

    # Gather log probabilities of the actual tokens
    # For each position, get log P(actual_token)
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # [batch, seq_len-1]

    # Mask out padding and prompt tokens (labels == -100)
    mask = (shift_labels != -100).float()

    # Sum log probabilities over sequence (only non-masked tokens)
    # This gives us log P(sequence)
    sequence_log_prob = (token_log_probs * mask).sum(dim=-1)  # [batch]

    return sequence_log_prob


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the DPO loss.

    The DPO loss encourages the model to assign higher probability to chosen
    responses than rejected responses, while staying close to the reference model.

    Mathematical formulation:
        loss = -log(σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x)
                           - log π_ref(y_w|x) + log π_ref(y_l|x))))

    Where:
        - π_θ: Current model (being trained)
        - π_ref: Reference model (frozen)
        - y_w: Chosen (winning) response
        - y_l: Rejected (losing) response
        - x: Prompt
        - β: Temperature parameter
        - σ: Sigmoid function

    Intuition:
        - If model prefers chosen MORE than reference does → reward increases
        - If model prefers chosen LESS than reference does → penalty
        - Beta controls how strongly to enforce preferences

    Args:
        policy_chosen_logps: log π_θ(chosen) [batch]
        policy_rejected_logps: log π_θ(rejected) [batch]
        reference_chosen_logps: log π_ref(chosen) [batch]
        reference_rejected_logps: log π_ref(rejected) [batch]
        beta: Temperature parameter

    Returns:
        loss: DPO loss (scalar)
        metrics: Dict with useful metrics for logging
    """
    # Compute log ratios (implicit rewards)
    # This measures how much the model prefers chosen over rejected
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    # Compute the DPO logits
    # Positive = policy prefers chosen more than reference (good!)
    # Negative = policy prefers chosen less than reference (bad!)
    logits = beta * (policy_logratios - reference_logratios)

    # DPO loss: negative log-likelihood
    # We want high probability that chosen is better than rejected
    # sigmoid(logits) = P(chosen is better)
    # -log(sigmoid(logits)) = cross-entropy loss
    loss = -F.logsigmoid(logits).mean()

    # Compute metrics for logging
    with torch.no_grad():
        # Implicit reward for chosen vs rejected
        rewards_chosen = beta * (policy_chosen_logps - reference_chosen_logps)
        rewards_rejected = beta * (policy_rejected_logps - reference_rejected_logps)
        reward_margin = (rewards_chosen - rewards_rejected).mean().item()

        # Accuracy: how often does model prefer chosen over rejected?
        accuracy = (policy_logratios > 0).float().mean().item()

    metrics = {
        'reward_margin': reward_margin,
        'accuracy': accuracy,
        'chosen_reward': rewards_chosen.mean().item(),
        'rejected_reward': rewards_rejected.mean().item(),
    }

    return loss, metrics


# =============================================================================
# Training Loop
# =============================================================================

def get_lr(step: int, config: DPOConfig) -> float:
    """
    Linear warmup + constant learning rate.

    DPO typically uses a very low, constant learning rate since the model
    is already pretrained. We just add a short warmup for stability.
    """
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    return config.learning_rate


def train_dpo(config: DPOConfig):
    """
    Main DPO training function.

    This implements the complete DPO algorithm from scratch, with detailed
    comments explaining each step.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =========================================================================
    # 1. LOAD MODEL
    # =========================================================================

    print(f"\nLoading {config.model_type} model...")

    # Load pretrained model
    model = GPT.from_pretrained(config.model_type, weights_path=config.weights_path)
    model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")

    # =========================================================================
    # 2. CREATE REFERENCE MODEL
    # =========================================================================

    print("\nCreating reference model (frozen copy)...")

    # The reference model is a frozen copy of the initial model
    # It's CRITICAL for DPO to prevent the model from drifting too far
    model_ref = copy.deepcopy(model)
    model_ref.eval()

    # Freeze all parameters (no gradients needed)
    for param in model_ref.parameters():
        param.requires_grad = False

    print(f"Reference model created and frozen")
    print(f"Memory: Model (~{total_params*4/1024**2:.0f}MB) + Reference (~{total_params*4/1024**2:.0f}MB) = ~{total_params*8/1024**2:.0f}MB")

    # =========================================================================
    # 3. LOAD DATA
    # =========================================================================

    print(f"\nLoading preference data from {config.data_path}...")

    # We need a tokenizer for the dataset
    # GPT-2 uses tiktoken (GPT-2 tokenizer)
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
    except ImportError:
        print("Warning: tiktoken not found, using simple fallback tokenizer")
        # Fallback: very simple character-level tokenizer
        class SimpleTokenizer:
            def encode(self, text):
                return [ord(c) for c in text[:1000]]  # Limit length
            def decode(self, tokens):
                return ''.join(chr(t) for t in tokens if t < 128)
        tokenizer = SimpleTokenizer()

    # Load dataset
    full_dataset = PreferenceDataset(
        config.data_path,
        tokenizer,
        config.max_prompt_length,
        config.max_length,
    )

    # Split into train/val
    val_size = int(len(full_dataset) * config.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # =========================================================================
    # 4. SETUP OPTIMIZER
    # =========================================================================

    print(f"\nSetting up optimizer...")

    # Use AdamW with very low learning rate
    # DPO is sensitive to LR - too high causes instability
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    print(f"Optimizer: AdamW")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Beta (DPO temperature): {config.beta}")

    # =========================================================================
    # 5. TRAINING LOOP
    # =========================================================================

    print("\n" + "="*70)
    print("STARTING DPO TRAINING")
    print("="*70)
    print(f"Max steps: {config.max_steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.grad_accum_steps}")
    print(f"Effective batch size: {config.batch_size * config.grad_accum_steps}")
    print("="*70 + "\n")

    os.makedirs(config.output_dir, exist_ok=True)

    step = 0
    data_iter = iter(train_loader)

    while step < config.max_steps:
        t0 = time.time()

        # Zero gradients
        optimizer.zero_grad()

        # Gradient accumulation
        loss_accum = 0.0
        metrics_accum = {
            'reward_margin': 0.0,
            'accuracy': 0.0,
            'chosen_reward': 0.0,
            'rejected_reward': 0.0,
        }

        for micro_step in range(config.grad_accum_steps):
            # Get batch (with wraparound)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Move to device
            chosen_input_ids = batch['chosen_input_ids'].to(device)
            chosen_labels = batch['chosen_labels'].to(device)
            rejected_input_ids = batch['rejected_input_ids'].to(device)
            rejected_labels = batch['rejected_labels'].to(device)

            # ================================================================
            # FORWARD PASS: POLICY MODEL (trainable)
            # ================================================================

            # Compute log P(chosen | prompt) under current model
            policy_chosen_logps = compute_sequence_log_probs(
                model,
                chosen_input_ids,
                chosen_labels,
            )

            # Compute log P(rejected | prompt) under current model
            policy_rejected_logps = compute_sequence_log_probs(
                model,
                rejected_input_ids,
                rejected_labels,
            )

            # ================================================================
            # FORWARD PASS: REFERENCE MODEL (frozen)
            # ================================================================

            with torch.no_grad():
                # Compute log P(chosen | prompt) under reference model
                reference_chosen_logps = compute_sequence_log_probs(
                    model_ref,
                    chosen_input_ids,
                    chosen_labels,
                )

                # Compute log P(rejected | prompt) under reference model
                reference_rejected_logps = compute_sequence_log_probs(
                    model_ref,
                    rejected_input_ids,
                    rejected_labels,
                )

            # ================================================================
            # COMPUTE DPO LOSS
            # ================================================================

            loss, metrics = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=config.beta,
            )

            # Scale loss for gradient accumulation
            loss = loss / config.grad_accum_steps
            loss_accum += loss.detach()

            # Accumulate metrics
            for key in metrics_accum:
                metrics_accum[key] += metrics[key] / config.grad_accum_steps

            # ================================================================
            # BACKWARD PASS
            # ================================================================

            loss.backward()

        # Gradient clipping (important for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update learning rate
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Optimizer step
        optimizer.step()

        # Timing
        t1 = time.time()
        dt = (t1 - t0) * 1000  # milliseconds

        # Logging
        if step % config.log_interval == 0:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | "
                  f"reward_margin: {metrics_accum['reward_margin']:+.4f} | "
                  f"accuracy: {metrics_accum['accuracy']:.3f} | "
                  f"lr: {lr:.2e} | dt: {dt:.2f}ms")

        # Validation
        if step % config.eval_interval == 0 and step > 0:
            print("\n" + "="*70)
            print(f"VALIDATION AT STEP {step}")
            print("="*70)

            model.eval()
            val_loss = 0.0
            val_metrics = {
                'reward_margin': 0.0,
                'accuracy': 0.0,
                'chosen_reward': 0.0,
                'rejected_reward': 0.0,
            }

            with torch.no_grad():
                for val_batch in val_loader:
                    chosen_input_ids = val_batch['chosen_input_ids'].to(device)
                    chosen_labels = val_batch['chosen_labels'].to(device)
                    rejected_input_ids = val_batch['rejected_input_ids'].to(device)
                    rejected_labels = val_batch['rejected_labels'].to(device)

                    # Policy model
                    policy_chosen_logps = compute_sequence_log_probs(
                        model, chosen_input_ids, chosen_labels
                    )
                    policy_rejected_logps = compute_sequence_log_probs(
                        model, rejected_input_ids, rejected_labels
                    )

                    # Reference model
                    reference_chosen_logps = compute_sequence_log_probs(
                        model_ref, chosen_input_ids, chosen_labels
                    )
                    reference_rejected_logps = compute_sequence_log_probs(
                        model_ref, rejected_input_ids, rejected_labels
                    )

                    # Loss
                    loss, metrics = dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        reference_chosen_logps,
                        reference_rejected_logps,
                        beta=config.beta,
                    )

                    val_loss += loss.item()
                    for key in val_metrics:
                        val_metrics[key] += metrics[key]

            val_loss /= len(val_loader)
            for key in val_metrics:
                val_metrics[key] /= len(val_loader)

            print(f"val_loss: {val_loss:.6f}")
            print(f"val_reward_margin: {val_metrics['reward_margin']:+.4f}")
            print(f"val_accuracy: {val_metrics['accuracy']:.3f}")
            print("="*70 + "\n")

            model.train()

        # Checkpointing
        if step % config.save_interval == 0 and step > 0:
            checkpoint_path = os.path.join(config.output_dir, f"dpo_step_{step:06d}.pt")

            print(f"Saving checkpoint to {checkpoint_path}...")

            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
            }, checkpoint_path)

            print(f"✓ Saved checkpoint")

        step += 1

    # =========================================================================
    # 6. SAVE FINAL MODEL
    # =========================================================================

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

    final_path = os.path.join(config.output_dir, "dpo_final.pt")

    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
    }, final_path)

    print(f"\nFinal model saved to: {final_path}")
    print(f"\nYour model is now aligned with human preferences!")
    print(f"Use this model for generation to see the improved responses.")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 with DPO (from scratch)")

    # Model
    parser.add_argument("--model", type=str, default="gpt2",
                        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                        help="Model size")
    parser.add_argument("--weights", type=str, default="data/gpt2_weights.pt",
                        help="Path to pretrained weights")

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to preference data (JSONL format)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split fraction")

    # DPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO temperature parameter (higher = stronger preference)")

    # Training
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-7,
                        help="Learning rate (very low for stability)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Training steps")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")

    # Logging
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--eval-interval", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--save-interval", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--output", type=str, default="dpo_checkpoints",
                        help="Output directory")

    args = parser.parse_args()

    # Create config
    config = DPOConfig(
        model_type=args.model,
        weights_path=args.weights,
        data_path=args.data,
        val_split=args.val_split,
        beta=args.beta,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.steps,
        grad_accum_steps=args.grad_accum,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output,
    )

    # Train!
    train_dpo(config)


if __name__ == "__main__":
    main()
