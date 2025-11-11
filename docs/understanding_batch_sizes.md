# Understanding Batch Sizes: The Complete Picture

This document explains the **why** behind batch size choices in deep learning, not just the formulas.

## The Core Problem: Memory vs Optimization

Training a neural network requires two competing needs:

1. **Optimization wants large batches** (better gradient estimates)
2. **Hardware has limited memory** (can't fit large batches)

Gradient accumulation is the solution that lets us have both.

---

## The Three Batch Size Concepts

### 1. **Total Batch Size** (What the optimizer "sees")

**Definition**: Total number of tokens processed before updating weights

**Purpose**: Determines the **quality of the gradient estimate**

**Why it matters**:
```
Small batch (1-1000 tokens):
- Gradient is computed from few examples
- Very noisy - might point in wrong direction
- Like asking 1 person for directions: unreliable

Large batch (32K-64K tokens):
- Gradient averaged over many examples
- Smoother, more reliable
- Like asking 1000 people: wisdom of crowds

Too large (>1M tokens):
- Diminishing returns (not much smoother)
- Slower training (fewer weight updates)
- May hurt generalization
```

**The math**:
```
Total Batch Size = micro_batch_size × seq_len × grad_accum_steps × num_gpus

Example:
32768 = 16 × 256 × 8 × 1
  ↑      ↑    ↑    ↑   ↑
total  micro seq  accum GPUs
       batch  len  steps
```

**How to choose**:
- Look at papers for similar-sized models
- GPT-2 (124M params): 32K-64K tokens/step
- BERT (110M params): 256K tokens/step
- LLaMA (7B params): 4M tokens/step
- Rule of thumb: larger models need larger batches

---

### 2. **Micro Batch Size** (What fits in memory)

**Definition**: Number of sequences processed in a single forward/backward pass

**Purpose**: Maximum size that fits in **GPU memory** without OOM

**Why it matters**:
```
During training, GPU memory holds:
1. Model weights (fixed size)
2. Activations (grows with batch size!)
3. Gradients (same size as weights)
4. Optimizer state (same size as weights)

Activations are the killer:
- Store every layer's output for backprop
- Batch size 1:   ~2GB activations
- Batch size 16:  ~32GB activations (OOM!)
- Batch size 32:  ~64GB activations (definitely OOM!)

So micro_batch_size is LIMITED BY HARDWARE.
```

**How to find it**:
```python
# Binary search for max batch size that fits
for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    try:
        x = torch.randint(0, 50257, (batch_size, seq_len)).cuda()
        model.cuda()
        logits, loss = model(x, x)
        loss.backward()
        print(f"✓ batch_size={batch_size} fits")
    except RuntimeError:
        print(f"✗ batch_size={batch_size} OOM!")
        break
```

**Why not just use micro_batch_size = 1?**
- GPUs are parallel processors
- Batch size 1 wastes 90%+ of GPU cores (underutilized)
- Batch size 16 keeps GPU busy (efficient)
- **Goal**: Largest batch that fits = best hardware utilization

---

### 3. **Gradient Accumulation Steps** (The bridge between the two)

**Definition**: Number of micro-batches to accumulate before updating weights

**Purpose**: Simulate a large batch when you can't fit it in memory

**Why it works**:

The key insight is that **gradients are additive**:

```
Option A: Process 8 examples at once
  gradient = ∂loss/∂weights for all 8 examples

Option B: Process 1 example, 8 times
  grad1 = ∂loss/∂weights for example 1
  grad2 = ∂loss/∂weights for example 2
  ...
  grad8 = ∂loss/∂weights for example 8
  total_gradient = grad1 + grad2 + ... + grad8

RESULT: Option A and Option B give the SAME gradient!
```

**The algorithm**:
```python
optimizer.zero_grad()  # Start fresh

# Accumulation loop
for micro_step in range(grad_accum_steps):
    x, y = get_batch(micro_batch_size)
    loss = model(x, y)
    loss = loss / grad_accum_steps  # Important: divide by N
    loss.backward()  # Accumulates gradients

optimizer.step()  # Update weights with accumulated gradients
```

**Why divide loss by grad_accum_steps?**

Without division:
```
Step 1: loss=2.5, backward() → gradient gets +2.5
Step 2: loss=2.3, backward() → gradient gets +2.3
Step 3: loss=2.4, backward() → gradient gets +2.4
Total gradient = 7.2 (TOO LARGE!)
```

With division:
```
Step 1: loss=2.5/3=0.83, backward() → gradient gets +0.83
Step 2: loss=2.3/3=0.77, backward() → gradient gets +0.77
Step 3: loss=2.4/3=0.80, backward() → gradient gets +0.80
Total gradient = 2.4 (average, correct!)
```

**How to calculate**:
```python
grad_accum_steps = total_batch_size / (micro_batch_size × seq_len × num_gpus)

Example calculation:
- I want total_batch_size = 32768 (from GPT-2 paper)
- My GPU can fit micro_batch_size = 16
- I'm using seq_len = 256
- I have num_gpus = 1

grad_accum_steps = 32768 / (16 × 256 × 1)
                 = 32768 / 4096
                 = 8

So I do 8 forward/backward passes before optimizer.step()
```

---

## How They All Relate: The Complete Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    ONE OPTIMIZATION STEP                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Micro-batch 1│  │ Micro-batch 2│  │ Micro-batch 8│     │
│  │   (16×256)   │  │   (16×256)   │...│   (16×256)   │     │
│  │              │  │              │  │              │     │
│  │  Forward +   │  │  Forward +   │  │  Forward +   │     │
│  │  Backward    │  │  Backward    │  │  Backward    │     │
│  │              │  │              │     │              │     │
│  │ Gradients    │  │ Gradients    │  │ Gradients    │     │
│  │ accumulated  │  │ accumulated  │  │ accumulated  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│         └─────────────────┴─────────────────┘              │
│                           │                                 │
│                    Sum all gradients                        │
│                   (8 × 16 × 256 = 32768 tokens)            │
│                           │                                 │
│                      optimizer.step()                       │
│                   (Update weights ONCE)                     │
└─────────────────────────────────────────────────────────────┘
```

**Key insights**:

1. **micro_batch_size** = constrained by GPU memory (hardware limit)
2. **total_batch_size** = chosen from research/literature (optimization requirement)
3. **grad_accum_steps** = calculated to bridge the gap between them

**The relationship**:
```
Can't fit large batch → Use small micro batches
Can't train well with small batch → Accumulate gradients
Result → Get benefits of large batch with memory of small batch
```

---

## Real-World Example: Deriving From Scratch

Let's say you want to train a 50M parameter model on an RTX 3060 (12GB VRAM).

### Step 1: Find micro_batch_size (hardware constraint)

Run memory test:
```python
# Test increasing batch sizes
for bs in [1, 2, 4, 8, 16, 32]:
    try:
        test_forward_backward(batch_size=bs, seq_len=512)
        print(f"✓ {bs} works")
    except RuntimeError:
        print(f"✗ {bs} OOM")
        break
```

Result: `micro_batch_size = 8` fits, 16 OOMs

### Step 2: Choose total_batch_size (from research)

Search papers for 50M parameter models:
- Found similar models use 16K-32K tokens/step
- Choose `total_batch_size = 16384`

### Step 3: Choose seq_len (task requirement)

- Your task needs decent context
- Choose `seq_len = 512`

### Step 4: Calculate grad_accum_steps

```python
grad_accum_steps = total_batch_size / (micro_batch_size × seq_len × num_gpus)
                 = 16384 / (8 × 512 × 1)
                 = 16384 / 4096
                 = 4
```

### Step 5: Verify it makes sense

```
✓ grad_accum_steps = 4 (reasonable, 2-64 is typical)
✓ micro_batch_size = 8 (fits in memory)
✓ total_batch_size = 16384 (matches literature)
✓ Memory efficient: 8 examples at once, not 128
✓ Optimization effective: 16K tokens per update
```

---

## Why This Matters: The Trade-offs

### Scenario A: No gradient accumulation
```python
# Trying to fit entire batch in memory
micro_batch_size = 128  # Want this for optimization
seq_len = 256
# Total tokens = 128 × 256 = 32768 ✓

Problem: OOM! Can't fit 128 sequences in 12GB VRAM
```

### Scenario B: Small batch, no accumulation
```python
# What fits in memory
micro_batch_size = 8
seq_len = 256
# Total tokens = 8 × 256 = 2048

Problem: Only 2K tokens per step
- Very noisy gradients
- Unstable training
- Poor convergence
```

### Scenario C: Gradient accumulation (the solution)
```python
# What fits in memory
micro_batch_size = 8
seq_len = 256
grad_accum_steps = 16

# Effective total = 8 × 256 × 16 = 32768 ✓

Benefits:
✓ Fits in memory (only 8 sequences at once)
✓ Same optimization as large batch (32K tokens)
✓ Stable training
✓ No hardware upgrade needed
```

---

## Common Questions

### Q: Why not always use grad_accum_steps = 1?

**A**: You'd be limited to whatever fits in memory, which is usually too small for good optimization.

### Q: Why not use huge grad_accum_steps (like 1000)?

**A**: Diminishing returns + slower:
- Gradients don't get much better after ~32K-64K tokens
- Each weight update takes longer (1000 forward passes!)
- Fewer updates per hour = slower learning

### Q: Does gradient accumulation have any downsides?

**A**: Yes, one subtle issue:
- **BatchNorm** doesn't work correctly (computes stats per micro-batch)
- **Solution**: Don't use BatchNorm with gradient accumulation
- GPT-2 uses LayerNorm instead (no problem)

### Q: How does this relate to learning rate?

**A**: Larger total batch size requires larger learning rate:
```python
# Linear scaling rule
new_lr = base_lr × (new_batch_size / base_batch_size)

Example:
- Base: batch=16K, lr=3e-4
- New:  batch=64K (4× larger)
- New LR: 3e-4 × 4 = 1.2e-3
```

---

## Summary: The Big Picture

**The fundamental constraint**:
```
Good optimization needs:  32K-64K tokens per weight update
GPU memory allows:        4K-8K tokens at once
```

**The solution**:
```
Gradient accumulation = process in chunks, combine results
- Optimizer thinks it's seeing 32K tokens
- GPU only holds 4K tokens at once
- Everyone's happy!
```

**How to derive for your project**:
1. **Look up** total_batch_size for similar models (papers)
2. **Measure** max micro_batch_size on your GPU (test)
3. **Choose** seq_len based on task needs
4. **Calculate** grad_accum_steps = total / (micro × seq × gpus)
5. **Verify** grad_accum_steps is reasonable (2-64)

**The key relationships**:
```
total_batch_size = micro_batch_size × seq_len × grad_accum_steps × num_gpus
     ↑                   ↑                ↑            ↑               ↑
optimization      hardware         calculated    distributed
requirement       constraint       to bridge     training
(from papers)     (from test)      the gap      (if available)
```
