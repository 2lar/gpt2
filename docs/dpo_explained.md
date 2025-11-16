# DPO (Direct Preference Optimization) Explained

This document explains what DPO is, how it works, and what's actually happening inside `DPOTrainer`.

---

## Table of Contents

1. [What is DPO?](#what-is-dpo)
2. [The Math Behind DPO](#the-math-behind-dpo)
3. [DPO vs RLHF](#dpo-vs-rlhf)
4. [Custom Implementation](#custom-implementation)
5. [HuggingFace DPOTrainer vs Custom](#huggingface-vs-custom)
6. [When to Use Custom DPO](#when-to-use-custom)

---

## What is DPO?

**DPO (Direct Preference Optimization)** is a method for aligning language models to human preferences without reinforcement learning.

### The Problem

You have:
- A pretrained model (e.g., GPT-2, Llama)
- Preference data: pairs of (chosen, rejected) responses for the same prompt
- Goal: Make the model prefer "chosen" responses over "rejected" ones

**Traditional approach (RLHF):**
```
1. Train reward model on preferences (complex)
2. Use PPO to optimize policy against reward (very complex)
3. Requires: 2 models, reward modeling, RL infrastructure
```

**DPO approach:**
```
1. Train directly on preferences (simple!)
2. No reward model, no RL
3. Requires: Just your model + preference data
```

---

## The Math Behind DPO

### Preference Data

You have pairs of completions for the same prompt:

```python
{
    "prompt": "Explain machine learning",
    "chosen": "Machine learning is a subset of AI that enables...",   # Better
    "rejected": "Machine learning is when computers learn stuff.",     # Worse
}
```

### The DPO Loss Function

The core idea: Increase probability of chosen responses, decrease probability of rejected responses, while staying close to the original model.

**Loss formula:**

```
L_DPO = -log(σ(β * (log π_θ(y_w | x) - log π_θ(y_l | x) - log π_ref(y_w | x) + log π_ref(y_l | x))))

Where:
- π_θ: Current model being trained
- π_ref: Reference model (frozen copy of initial model)
- y_w: Chosen (winning) response
- y_l: Rejected (losing) response
- x: Prompt
- β: Temperature parameter (controls how much to prefer chosen over rejected)
- σ: Sigmoid function
```

**In simpler terms:**

1. Compute log-probability of chosen response under current model: `log π_θ(y_w | x)`
2. Compute log-probability of rejected response under current model: `log π_θ(y_l | x)`
3. Compute the same for reference model (frozen): `log π_ref(y_w | x)`, `log π_ref(y_l | x)`
4. Calculate difference (implicit reward): `log π_θ(y_w) - log π_θ(y_l) - log π_ref(y_w) + log π_ref(y_l)`
5. Scale by β (temperature)
6. Apply sigmoid and take negative log

**What this achieves:**
- Maximizes likelihood of chosen responses
- Minimizes likelihood of rejected responses
- Stays close to reference model (prevents divergence)

### Why the Reference Model?

The reference model `π_ref` is a frozen copy of your model before DPO training.

**Purpose:**
- Prevents the model from drifting too far from its original behavior
- Acts as a regularizer (KL divergence penalty)
- Ensures the model doesn't "forget" what it learned during pretraining

**In practice:**
```python
# Before DPO
model_ref = copy.deepcopy(model)  # Freeze a copy
model_ref.eval()
model_ref.requires_grad_(False)

# During DPO training
# Compare model (trainable) to model_ref (frozen)
```

---

## DPO vs RLHF

### Traditional RLHF (Reinforcement Learning from Human Feedback)

```
Phase 1: Train Reward Model
├─ Input: Preference pairs (chosen, rejected)
├─ Train a classifier to predict which response is better
└─ Output: Reward model r(x, y) that scores responses

Phase 2: RL Training (PPO)
├─ Use reward model to score model outputs
├─ Run PPO (Proximal Policy Optimization) to maximize reward
├─ Requires: Value network, policy network, RL infrastructure
└─ Complex and unstable!

Total: 2 phases, 2+ models, RL expertise required
```

### DPO (Direct Preference Optimization)

```
Single Phase: Direct Training
├─ Input: Preference pairs (chosen, rejected)
├─ Train model directly with DPO loss
├─ No reward model needed
└─ No RL needed

Total: 1 phase, 1 model, standard supervised learning
```

### Comparison Table

| Aspect | RLHF (PPO) | DPO |
|--------|------------|-----|
| **Phases** | 2 (reward model + RL) | 1 (direct optimization) |
| **Models** | 3+ (policy, value, reward) | 2 (model, reference) |
| **Complexity** | Very high (RL) | Low (supervised learning) |
| **Stability** | Unstable (RL training) | Stable (standard loss) |
| **Speed** | Slow (RL iterations) | Fast (standard backprop) |
| **Performance** | Slightly better (in some cases) | Comparable (often equal) |
| **Used by** | OpenAI (ChatGPT), Anthropic | Most open models (Llama, Mistral) |

**Reality:** DPO has largely replaced RLHF for open-source models because it's simpler and works just as well.

---

## Custom Implementation

Let me show you what `DPOTrainer` is doing under the hood:

### Step 1: The DPO Loss Function

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # log π_θ(y_w | x)
    policy_rejected_logps: torch.Tensor,  # log π_θ(y_l | x)
    reference_chosen_logps: torch.Tensor, # log π_ref(y_w | x)
    reference_rejected_logps: torch.Tensor, # log π_ref(y_l | x)
    beta: float = 0.1,  # Temperature parameter
) -> torch.Tensor:
    """
    Compute the DPO loss for a batch of preference pairs.

    The loss encourages the model to:
    1. Increase probability of chosen responses
    2. Decrease probability of rejected responses
    3. Stay close to the reference model (KL penalty)

    Args:
        policy_chosen_logps: Log probabilities of chosen responses under current model
        policy_rejected_logps: Log probabilities of rejected responses under current model
        reference_chosen_logps: Log probabilities of chosen responses under reference model
        reference_rejected_logps: Log probabilities of rejected responses under reference model
        beta: Temperature parameter (higher = stronger preference for chosen)

    Returns:
        DPO loss (scalar)
    """
    # Compute log ratios (implicit rewards)
    # This measures: "How much does the current model prefer chosen over rejected,
    #                 compared to the reference model?"
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    # Compute logits for the preference (scaled difference)
    # Positive = model prefers chosen more than reference does (good!)
    # Negative = model prefers chosen less than reference does (bad!)
    logits = beta * (policy_logratios - reference_logratios)

    # DPO loss: negative log-likelihood of choosing the chosen response
    # sigmoid(logits) = probability that chosen is better
    # -log(sigmoid(logits)) = cross-entropy loss
    loss = -F.logsigmoid(logits).mean()

    return loss


def compute_log_probability(model, input_ids, labels):
    """
    Compute log probability of a sequence under the model.

    Args:
        model: Language model
        input_ids: Input token IDs [batch, seq_len]
        labels: Target token IDs [batch, seq_len] (same as input_ids shifted)

    Returns:
        Log probability of the sequence (summed over tokens)
    """
    # Forward pass
    outputs = model(input_ids)
    logits = outputs.logits  # [batch, seq_len, vocab_size]

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]

    # Gather log probabilities of actual tokens
    # log_probs[i, j, labels[i, j]] = log probability of token at position (i, j)
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)  # [batch, seq_len]

    # Sum over sequence length (total log probability)
    # Ignore padding tokens (assuming labels == -100 for padding)
    mask = (labels != -100).float()
    sequence_log_prob = (token_log_probs * mask).sum(dim=-1)  # [batch]

    return sequence_log_prob
```

### Step 2: The Training Loop

```python
def train_dpo_step(
    model,           # Trainable model
    model_ref,       # Reference model (frozen)
    batch,           # Batch of preference data
    optimizer,
    beta=0.1,
):
    """
    Single DPO training step.

    Args:
        model: Current model being trained
        model_ref: Reference model (frozen copy)
        batch: Dict with keys 'prompt_ids', 'chosen_ids', 'rejected_ids'
        optimizer: Optimizer
        beta: DPO temperature

    Returns:
        Loss value
    """
    # Unpack batch
    prompt_ids = batch['prompt_ids']          # [batch, prompt_len]
    chosen_ids = batch['chosen_ids']          # [batch, chosen_len]
    rejected_ids = batch['rejected_ids']      # [batch, rejected_len]

    # Concatenate prompt + response
    chosen_input_ids = torch.cat([prompt_ids, chosen_ids], dim=1)
    rejected_input_ids = torch.cat([prompt_ids, rejected_ids], dim=1)

    # Create labels (only compute loss on response tokens, not prompt)
    chosen_labels = chosen_input_ids.clone()
    chosen_labels[:, :prompt_ids.shape[1]] = -100  # Ignore prompt tokens

    rejected_labels = rejected_input_ids.clone()
    rejected_labels[:, :prompt_ids.shape[1]] = -100

    # ========================================================================
    # Forward pass: Current model (trainable)
    # ========================================================================

    policy_chosen_logps = compute_log_probability(model, chosen_input_ids, chosen_labels)
    policy_rejected_logps = compute_log_probability(model, rejected_input_ids, rejected_labels)

    # ========================================================================
    # Forward pass: Reference model (frozen)
    # ========================================================================

    with torch.no_grad():
        reference_chosen_logps = compute_log_probability(model_ref, chosen_input_ids, chosen_labels)
        reference_rejected_logps = compute_log_probability(model_ref, rejected_input_ids, rejected_labels)

    # ========================================================================
    # Compute DPO loss
    # ========================================================================

    loss = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta=beta,
    )

    # ========================================================================
    # Backward pass and optimizer step
    # ========================================================================

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

### Step 3: Full Training Loop

```python
def train_dpo(
    model,
    train_dataset,
    tokenizer,
    num_epochs=1,
    batch_size=4,
    learning_rate=5e-7,
    beta=0.1,
):
    """
    Complete DPO training loop.

    This is what DPOTrainer does under the hood!
    """
    # Create reference model (frozen copy)
    import copy
    model_ref = copy.deepcopy(model)
    model_ref.eval()
    for param in model_ref.parameters():
        param.requires_grad = False

    print("Created reference model (frozen)")

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Setup dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for step, batch in enumerate(dataloader):
            # Train step
            loss = train_dpo_step(
                model,
                model_ref,
                batch,
                optimizer,
                beta=beta,
            )

            total_loss += loss

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} complete. Average loss: {avg_loss:.4f}")

    print("DPO training complete!")

    return model
```

---

## HuggingFace DPOTrainer vs Custom

### What HuggingFace DPOTrainer Does

```python
from trl import DPOTrainer

# This 5-line code...
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=preference_dataset,
    processing_class=tokenizer,
)
trainer.train()

# ...is equivalent to ~500 lines that do:
1. Create reference model (frozen copy)
2. Setup optimizer and scheduler
3. Setup data loader with preference pairs
4. For each batch:
   a. Tokenize prompt + chosen + rejected
   b. Forward pass through model (trainable)
   c. Forward pass through reference model (frozen)
   d. Compute DPO loss
   e. Backward pass
   f. Optimizer step
5. Logging and checkpointing
6. Evaluation on validation set
```

### The Full Picture

```python
class DPOTrainer:
    """
    What DPOTrainer actually does (simplified).
    """

    def __init__(self, model, args, train_dataset, processing_class):
        # 1. Create reference model
        self.model = model
        self.model_ref = copy.deepcopy(model)
        self.model_ref.eval()
        self.model_ref.requires_grad_(False)

        # 2. Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
        )

        # 3. Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps,
        )

        # 4. Setup dataloader
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
        )

        # 5. Store args
        self.args = args
        self.tokenizer = processing_class

    def train(self):
        """Main training loop."""
        self.model.train()

        for epoch in range(self.args.num_train_epochs):
            for batch in self.dataloader:
                # Tokenize preference pairs
                prompt_ids, chosen_ids, rejected_ids = self.prepare_batch(batch)

                # Compute log probabilities (policy model)
                policy_chosen_logps = self.compute_logps(
                    self.model, prompt_ids, chosen_ids
                )
                policy_rejected_logps = self.compute_logps(
                    self.model, prompt_ids, rejected_ids
                )

                # Compute log probabilities (reference model)
                with torch.no_grad():
                    ref_chosen_logps = self.compute_logps(
                        self.model_ref, prompt_ids, chosen_ids
                    )
                    ref_rejected_logps = self.compute_logps(
                        self.model_ref, prompt_ids, rejected_ids
                    )

                # DPO loss
                loss = self.dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=self.args.beta,
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Logging
                if step % self.args.logging_steps == 0:
                    self.log({"loss": loss.item()})

                # Checkpointing
                if step % self.args.save_steps == 0:
                    self.save_checkpoint()
```

---

## When to Use Custom DPO

### Use HuggingFace DPOTrainer When:

✅ **Standard DPO alignment** (most cases)
✅ **Production fine-tuning** of open models
✅ **Quick experiments** with preference data
✅ **Team lacks RL/alignment expertise**
✅ **Integrating with HuggingFace ecosystem**

**Example:** Fine-tuning Llama for chatbot
```python
from trl import DPOTrainer

# Just works!
trainer = DPOTrainer(model=model, train_dataset=prefs, ...)
trainer.train()
```

### Use Custom DPO When:

✅ **Research on DPO variants** (IPO, KTO, etc.)
✅ **Novel loss functions** (custom preferences)
✅ **Learning how DPO works** (educational)
✅ **Debugging alignment issues**
✅ **Extreme customization** (multi-reward, constraints)
✅ **Production at frontier labs** (OpenAI-scale)

**Example:** Researching a new DPO variant
```python
# Custom loss function
def my_custom_dpo_loss(policy_logps, ref_logps, beta):
    # Your novel idea here!
    ...

# Full control
for batch in dataloader:
    loss = my_custom_dpo_loss(...)
    loss.backward()
    optimizer.step()
```

---

## DPO Variants (Why You'd Need Custom)

Since DPO's introduction, researchers have proposed many variants. These require custom implementations:

### 1. IPO (Identity Preference Optimization)

Modified loss that removes the sigmoid:
```python
# DPO loss
loss = -log(sigmoid(β * (logits)))

# IPO loss
loss = (β * (logits) - 1/2)²
```

**When to use:** More stable training, less sensitive to β

### 2. KTO (Kahneman-Tversky Optimization)

Uses single examples instead of pairs:
```python
# Instead of (chosen, rejected) pairs
# Just labeled examples: (response, is_good)

loss = loss_good(y) if is_good else loss_bad(y)
```

**When to use:** When you only have thumbs-up/down, not pairs

### 3. ORPO (Odds Ratio Preference Optimization)

Combines SFT and DPO in one loss:
```python
loss = sft_loss + dpo_loss
```

**When to use:** Don't want separate SFT phase

### 4. Multi-Reward DPO

Multiple types of preferences:
```python
# Different β for different reward types
loss = dpo_loss(helpful_prefs, β=0.1) + dpo_loss(safe_prefs, β=0.5)
```

**When to use:** Multiple objectives (helpful + safe + harmless)

**To implement these, you need custom scripts!**

---

## Summary

### The Spectrum for DPO

```
HuggingFace DPOTrainer                           Custom DPO
│                                                           │
│  5 lines of code                                500 lines│
│  Black box                                    Full control│
│  Standard DPO                                    Variants │
│  Production (80%)                            Research (20%)│
└─────────────────────────────────────────────────────────┘
```

### When to Use What

| Your Goal | Use This |
|-----------|----------|
| Fine-tune Llama for chat | HuggingFace DPOTrainer |
| Production alignment | HuggingFace DPOTrainer |
| Learn how DPO works | Custom implementation |
| Research DPO variants | Custom implementation |
| Multi-reward alignment | Custom implementation |
| Frontier lab scale | Custom implementation |

### The Reality

**Most companies:** Use HuggingFace DPOTrainer
- It just works
- Well-tested
- Easy to use

**Frontier labs:** Custom everything
- Full control
- Novel variants
- Extreme optimization

**You (learning):** Understand both!
- Use DPOTrainer for quick experiments
- Write custom DPO to understand how it works
- Best of both worlds

---

## Next Steps

1. **Read the custom implementation** above to understand DPO
2. **Try HuggingFace DPOTrainer** for quick experiments
3. **Implement custom DPO** for a toy problem
4. **Compare results** between standard DPO and variants

**Resources:**
- Original DPO paper: https://arxiv.org/abs/2305.18290
- HuggingFace TRL docs: https://huggingface.co/docs/trl/dpo_trainer
- DPO variants survey: https://arxiv.org/abs/2404.14367

Happy aligning! 🎯
