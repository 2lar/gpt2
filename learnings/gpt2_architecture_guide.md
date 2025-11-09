# GPT-2 Architecture Deep Dive

A comprehensive guide to the modular GPT-2 implementation in this repository.

---

## Table of Contents

1. [Overview](#overview)
2. [Package Structure](#package-structure)
3. [Configuration (`config.py`)](#configuration)
4. [Architecture Components](#architecture-components)
   - [Embeddings](#embeddings)
   - [Attention Mechanism](#attention-mechanism)
   - [MLP (Feed-Forward Network)](#mlp-feed-forward-network)
   - [Transformer Block](#transformer-block)
   - [Full GPT Model](#full-gpt-model)
5. [Data Loading](#data-loading)
6. [Training Pipeline](#training-pipeline)
7. [Parameter Counts](#parameter-counts)
8. [Data Flow Visualization](#data-flow-visualization)

---

## Overview

This is a **modular, production-ready implementation** of GPT-2 that improves upon the single-file reference implementation with:
- ✅ Clean separation of concerns
- ✅ Full dropout support (attention, residual, embedding)
- ✅ Configurable bias terms
- ✅ Type hints throughout
- ✅ Comprehensive documentation
- ✅ Distributed training support
- ✅ HuggingFace checkpoint compatibility

**Files:** `gpt2/config.py`, `gpt2/attention.py`, `gpt2/block.py`, `gpt2/model.py`, `gpt2/data.py`, `gpt2/evaluation.py`, `scripts/train.py`

---

## Package Structure

```
gpt2/
├── __init__.py          # Package exports
├── config.py            # GPTConfig dataclass
├── attention.py         # CausalSelfAttention
├── block.py             # Block (Transformer layer) + MLP
├── model.py             # GPT (main model)
├── data.py              # DataLoaderLite
└── evaluation.py        # Evaluation utilities

scripts/
├── train.py             # Training script with DDP support
├── generate.py          # Text generation
├── validate.py          # Validation vs HuggingFace
└── eval_hellaswag.py    # HellaSwag benchmark
```

---

## Configuration

**File:** `gpt2/config.py`

```python
@dataclass
class GPTConfig:
    block_size: int = 1024    # Max sequence length (context window)
    vocab_size: int = 50257   # Number of tokens in vocabulary
    n_layer: int = 12         # Number of Transformer blocks (DEPTH)
    n_head: int = 12          # Number of attention heads
    n_embd: int = 768         # Embedding dimension (WIDTH)
    dropout: float = 0.0      # Dropout probability
    bias: bool = True         # Use bias in Linear/LayerNorm
```

### GPT-2 Model Variants

| Model | n_layer | n_head | n_embd | Parameters |
|-------|---------|--------|--------|------------|
| **GPT-2 Small** | 12 | 12 | 768 | 124M |
| **GPT-2 Medium** | 24 | 16 | 1024 | 350M |
| **GPT-2 Large** | 36 | 20 | 1280 | 774M |
| **GPT-2 XL** | 48 | 25 | 1600 | 1558M |

### Key Dimensions Explained

#### `n_embd` (Embedding Dimension) - The "Width"
- **What it is:** The fundamental dimension of the "residual stream"
- **Where it's used:**
  - Token embeddings: `(vocab_size, n_embd)`
  - Position embeddings: `(block_size, n_embd)`
  - Hidden states throughout: `(B, T, n_embd)`
  - Attention input/output: `n_embd → n_embd`
  - MLP input/output: `n_embd → 4*n_embd → n_embd`
  - LayerNorm: operates on `n_embd` dimension

**Think of it as:** The width of the information highway

#### `n_layer` (Number of Layers) - The "Depth"
- **What it is:** Number of Transformer blocks stacked sequentially
- **Each layer contains:**
  - 1 LayerNorm
  - 1 CausalSelfAttention (2 Linear layers)
  - 1 LayerNorm
  - 1 MLP (2 Linear layers)
  - Total: ~4 Linear layers per Transformer block

**Think of it as:** How many times information is processed

#### `n_head` (Number of Attention Heads)
- **What it is:** Parallel attention operations in multi-head attention
- **Constraint:** `n_embd` must be divisible by `n_head`
- **Head dimension:** `head_dim = n_embd // n_head`
  - GPT-2 Small: `768 // 12 = 64` dimensions per head

**Think of it as:** How many different attention patterns we learn

---

## Architecture Components

### Embeddings

**Location:** `gpt2/model.py` lines 50-55

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(vocab_size, n_embd),    # Token embeddings
    wpe = nn.Embedding(block_size, n_embd),    # Position embeddings
    drop = nn.Dropout(dropout),                 # Embedding dropout
    ...
))
```

#### How Embeddings Work

```
Input: Token IDs
  [15496, 11, 314, 1101, ...]  # Shape: (B, T)
       ↓
Token Embedding Lookup (wte)
  each ID → 768-dim vector
       ↓
  [[0.023, -0.145, ..., 0.891],   # Shape: (B, T, 768)
   [0.512, -0.023, ..., -0.234],
   ...]
       +
Position Embedding (wpe)
  position 0 → [0.012, 0.234, ...]
  position 1 → [-0.123, 0.456, ...]
  ...
       ↓
Combined: token + position
  [[0.035, 0.089, ..., 0.901],    # Shape: (B, T, 768)
   [0.389, 0.433, ..., -0.124],
   ...]
       ↓
Dropout (regularization)
       ↓
Ready for Transformer blocks
```

**Key Points:**
- Token embeddings: Learn what each token means
- Position embeddings: Learn where each token is (learned, not sinusoidal)
- Combined via simple addition (broadcasts across batch dimension)
- Dropout applied to combined embeddings

---

### Attention Mechanism

**File:** `gpt2/attention.py`

Multi-head causal self-attention with Flash Attention optimization.

#### Architecture

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        # Combined QKV projection (more efficient than 3 separate)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # Special initialization

        # Dropout
        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
```

#### Attention Flow

```
Input: x (B, T, 768)
       ↓
┌─────────────────────────────────────────┐
│ Step 1: Project to Q, K, V              │
│                                         │
│  x → Linear(768, 2304)                  │
│    = c_attn projection                  │
│       ↓                                 │
│  QKV tensor: (B, T, 2304)               │
│       ↓                                 │
│  Split into 3 pieces:                   │
│    Q: (B, T, 768)                       │
│    K: (B, T, 768)                       │
│    V: (B, T, 768)                       │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Step 2: Reshape for Multi-Head          │
│                                         │
│  Q: (B, T, 768)                         │
│    → (B, T, 12, 64)  # 12 heads, 64 dim │
│    → (B, 12, T, 64)  # heads to batch   │
│                                         │
│  K: (B, T, 768) → (B, 12, T, 64)        │
│  V: (B, T, 768) → (B, 12, T, 64)        │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Step 3: Scaled Dot-Product Attention    │
│                                         │
│  Attention(Q,K,V) =                     │
│    softmax(Q·K^T / √d_k) · V            │
│                                         │
│  Uses Flash Attention:                  │
│    F.scaled_dot_product_attention(      │
│      q, k, v,                           │
│      is_causal=True,  # Causal mask     │
│      dropout_p=dropout                  │
│    )                                    │
│                                         │
│  Causal mask ensures token i can only   │
│  attend to tokens ≤ i (no future peek)  │
│                                         │
│  Output: (B, 12, T, 64)                 │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Step 4: Concatenate Heads               │
│                                         │
│  (B, 12, T, 64)                         │
│    → (B, T, 12, 64)  # transpose        │
│    → (B, T, 768)     # reshape/concat   │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Step 5: Output Projection               │
│                                         │
│  (B, T, 768) → Linear(768, 768)         │
│              → Dropout                  │
│              → (B, T, 768)              │
└─────────────────────────────────────────┘
       ↓
Output: (B, T, 768)
```

#### What is Causal Masking?

Causal masking (autoregressive property) ensures each token can only attend to previous tokens:

```
For sequence: ["The", "cat", "sat", "on"]

Attention Matrix (what each token can see):
         The  cat  sat  on
    The   ✓    ✗    ✗    ✗     # "The" only sees itself
    cat   ✓    ✓    ✗    ✗     # "cat" sees "The" and "cat"
    sat   ✓    ✓    ✓    ✗     # "sat" sees first 3 tokens
    on    ✓    ✓    ✓    ✓     # "on" sees all tokens

This is a lower-triangular mask
```

**Why?** During training, this prevents the model from "cheating" by looking at future tokens.

---

### MLP (Feed-Forward Network)

**File:** `gpt2/block.py` lines 17-66

Position-wise feed-forward network applied after attention.

#### Architecture

```python
class MLP(nn.Module):
    def __init__(self, config):
        # Expand: n_embd → 4*n_embd
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=config.bias)

        # GELU activation (smoother than ReLU)
        self.gelu = nn.GELU(approximate='tanh')

        # Project back: 4*n_embd → n_embd
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.dropout = nn.Dropout(config.dropout)
```

#### MLP Flow

```
Input: (B, T, 768)
       ↓
┌──────────────────────┐
│ Expand (c_fc)        │
│ Linear: 768 → 3072   │  # 4× expansion
│ (B, T, 768)          │
│    → (B, T, 3072)    │
└──────────────────────┘
       ↓
┌──────────────────────┐
│ GELU Activation      │
│ Smooth non-linearity │
│ (B, T, 3072)         │
│    → (B, T, 3072)    │
└──────────────────────┘
       ↓
┌──────────────────────┐
│ Project (c_proj)     │
│ Linear: 3072 → 768   │  # Back to residual stream
│ (B, T, 3072)         │
│    → (B, T, 768)     │
└──────────────────────┘
       ↓
┌──────────────────────┐
│ Dropout              │
│ Regularization       │
└──────────────────────┘
       ↓
Output: (B, T, 768)
```

**Why 4× expansion?**
- Standard from original Transformer paper
- Gives model capacity to learn complex transformations
- Bottleneck architecture: expand → transform → compress

---

### Transformer Block

**File:** `gpt2/block.py` lines 68-127

The fundamental building block of GPT-2. Each "layer" is one Block.

#### Architecture

```python
class Block(nn.Module):
    def __init__(self, config):
        self.ln_1 = nn.LayerNorm(n_embd)           # Pre-attention norm
        self.attn = CausalSelfAttention(config)    # Multi-head attention
        self.ln_2 = nn.LayerNorm(n_embd)           # Pre-MLP norm
        self.mlp = MLP(config)                     # Feed-forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Attention + residual
        x = x + self.mlp(self.ln_2(x))   # MLP + residual
        return x
```

#### Block Flow (Pre-Norm Architecture)

```
Input: x (B, T, 768)
       ↓
┌─────────────────────────────────────┐
│ ATTENTION BLOCK                     │
│                                     │
│  x_norm = LayerNorm(x)              │  ← Normalize BEFORE attention
│       ↓                             │
│  attn_out = Attention(x_norm)       │  ← Multi-head attention
│       ↓                             │
│  x = x + attn_out                   │  ← Residual connection
│                                     │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ MLP BLOCK                           │
│                                     │
│  x_norm = LayerNorm(x)              │  ← Normalize BEFORE MLP
│       ↓                             │
│  mlp_out = MLP(x_norm)              │  ← Feed-forward
│       ↓                             │
│  x = x + mlp_out                    │  ← Residual connection
│                                     │
└─────────────────────────────────────┘
       ↓
Output: x (B, T, 768)
```

#### Pre-Norm vs Post-Norm

**This implementation uses Pre-Norm** (LayerNorm before sub-layers):
```
x = x + Attention(LayerNorm(x))  ✅ Pre-Norm (GPT-2, modern)
```

**Original Transformer used Post-Norm:**
```
x = LayerNorm(x + Attention(x))  ❌ Post-Norm (harder to train)
```

**Why Pre-Norm?**
- More stable gradient flow
- Easier to train deep networks
- Better performance empirically

#### Residual Connections

The `x = x + ...` pattern is crucial:
- Allows gradients to flow directly through the network
- Prevents vanishing gradients in deep networks
- Creates a "residual stream" that carries information

Think of it as:
- **Attention**: Communication between tokens (gather info from context)
- **MLP**: Computation on individual tokens (process the info)
- **Residual**: Main highway (ensures information isn't lost)

---

### Full GPT Model

**File:** `gpt2/model.py`

Assembles all components into the complete GPT-2 architecture.

#### Model Structure

```python
class GPT(nn.Module):
    def __init__(self, config):
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),     # Token embeddings
            wpe = nn.Embedding(block_size, n_embd),     # Position embeddings
            drop = nn.Dropout(dropout),                 # Embedding dropout
            h = nn.ModuleList([Block(config) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),                # Final layer norm
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share embeddings with output
        self.transformer.wte.weight = self.lm_head.weight
```

#### Complete Forward Pass

```
Input: Token IDs (B, T)
  Example: [15496, 11, 314, ...]
       ↓
┌────────────────────────────────────────────┐
│ EMBEDDING LAYER                            │
│                                            │
│  Token Embedding (wte)                     │
│    (B, T) → (B, T, 768)                    │
│       +                                    │
│  Position Embedding (wpe)                  │
│    (T,) → (T, 768) [broadcast to (B,T,768)]│
│       ↓                                    │
│  x = tok_emb + pos_emb  (B, T, 768)        │
│       ↓                                    │
│  Dropout(x)                                │
└────────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────────┐
│ TRANSFORMER BLOCK 1                        │
│                                            │
│  x = x + Attention(LayerNorm(x))           │
│  x = x + MLP(LayerNorm(x))                 │
│                                            │
│  (B, T, 768) → (B, T, 768)                 │
└────────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────────┐
│ TRANSFORMER BLOCK 2                        │
│  ...                                       │
└────────────────────────────────────────────┘
       ↓
       ... (blocks 3-11)
       ↓
┌────────────────────────────────────────────┐
│ TRANSFORMER BLOCK 12                       │
│  ...                                       │
└────────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────────┐
│ FINAL LAYER NORM                           │
│                                            │
│  x = LayerNorm(x)                          │
│  (B, T, 768) → (B, T, 768)                 │
└────────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────────┐
│ LANGUAGE MODEL HEAD                        │
│                                            │
│  logits = Linear(x, vocab_size)            │
│  (B, T, 768) → (B, T, 50257)               │
│                                            │
│  For each position, produces a score       │
│  for each possible next token              │
└────────────────────────────────────────────┘
       ↓
Output: Logits (B, T, 50257)
       ↓
┌────────────────────────────────────────────┐
│ LOSS CALCULATION (if training)             │
│                                            │
│  Cross-entropy loss between:               │
│    - Predicted logits: (B*T, 50257)        │
│    - Target tokens: (B*T,)                 │
│                                            │
│  Each position predicts the NEXT token     │
└────────────────────────────────────────────┘
       ↓
Loss: scalar value
```

#### Weight Tying

```python
self.transformer.wte.weight = self.lm_head.weight
```

**What this means:**
- Token embedding matrix: `(50257, 768)` - maps token IDs to vectors
- LM head matrix: `(768, 50257)` - maps vectors to token probabilities
- These are **the same matrix**, just transposed!

**Benefits:**
1. Reduces parameters significantly (~38M parameters saved)
2. Forces consistent token representation space
3. Improves performance empirically

#### Weight Initialization

Special initialization scheme from GPT-2 paper:

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        std = 0.02

        # Residual projections scaled by depth
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 * n_layer) ** -0.5  # Scale down for deep networks

        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
```

**Why scale residual layers?**
- In deep networks, residual contributions accumulate
- Without scaling, activations can explode
- Scaling by `1/√(2*n_layer)` keeps variance stable

---

## Data Loading

**File:** `gpt2/data.py`

Efficient data loader for pre-tokenized language modeling data.

### Architecture

```python
@dataclass
class LoaderConfig:
    batch_size: int          # Sequences per batch
    seq_len: int            # Tokens per sequence (T)
    process_rank: int       # For distributed training
    world_size: int         # Number of processes
    split: str              # "train" or "val"
    data_root: str          # Directory with .npy shards
    master_process: bool    # For logging

class DataLoaderLite:
    def __init__(self, cfg: LoaderConfig):
        # Find all shard files (e.g., train_00001.npy)
        self.shards = [list of .npy files]
        self.reset()

    def next_batch(self) -> (x, y):
        # Returns (inputs, targets) tensors
        ...
```

### How Data Loading Works

```
Disk: Pre-tokenized shards
  ├─ train_00001.npy  [millions of tokens]
  ├─ train_00002.npy
  └─ ...

       ↓ load_tokens()

Memory: Token tensor
  [15496, 11, 314, 1101, 257, 3303, ...]

       ↓ next_batch()

Extract chunk:
  Position: current_position
  Length: B * T + 1

  [tok₀, tok₁, tok₂, ..., tok_{B*T}]

       ↓ Split into input/target

  x (inputs):  [tok₀, tok₁, ..., tok_{B*T-1}]  → reshape to (B, T)
  y (targets): [tok₁, tok₂, ..., tok_{B*T}]    → reshape to (B, T)

  Each input token predicts the NEXT token

       ↓ Advance position

  current_position += B * T * world_size

  (In distributed training, this skips data used by other processes)
```

### Distributed Training Support

For 4 GPUs, batch_size=64, seq_len=1024:

```
Shard: [tok₀, tok₁, tok₂, ..., tok₁₀₀₀₀₀₀]

GPU 0: starts at position 0
  ├─ Reads tokens [0:65536]
  └─ Next read at position 262144 (65536 * 4)

GPU 1: starts at position 65536
  ├─ Reads tokens [65536:131072]
  └─ Next read at position 327680

GPU 2: starts at position 131072
  ├─ Reads tokens [131072:196608]
  └─ Next read at position 393216

GPU 3: starts at position 196608
  ├─ Reads tokens [196608:262144]
  └─ Next read at position 458752

Each GPU sees different, non-overlapping data!
```

### Infinite Iterator

```python
def __iter__(self):
    while True:
        yield self.next_batch()
```

**Why infinite?**
- Training loops typically run for a fixed number of steps, not epochs
- When reaching end of shard, automatically cycles to next shard
- Simplifies training code (no need to handle epoch boundaries)

---

## Training Pipeline

**File:** `scripts/train.py`

Complete training script with distributed data parallel support.

### Training Configuration

```python
@dataclass
class TrainingConfig:
    data_root: str = "edu_fineweb10B"
    total_batch_size: int = 524288      # ~0.5M tokens per update
    micro_batch_size: int = 64          # Sequences per forward pass
    seq_len: int = 1024                 # Tokens per sequence
    max_lr: float = 6e-4                # Peak learning rate
    min_lr_ratio: float = 0.1           # min_lr = max_lr * 0.1
    warmup_steps: int = 715             # Linear warmup
    max_steps: int = 19073              # ~1 epoch on 10B tokens
    weight_decay: float = 0.1           # AdamW weight decay
    ...
```

### Training Loop Structure

```
┌─────────────────────────────────────────────┐
│ Setup                                       │
├─────────────────────────────────────────────┤
│ • Initialize distributed training (DDP)     │
│ • Create model, move to device              │
│ • Create optimizer (AdamW)                  │
│ • Create data loaders (train/val)           │
└─────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────┐
│ Training Loop (max_steps iterations)        │
├─────────────────────────────────────────────┤
│                                             │
│  for step in range(max_steps):              │
│                                             │
│    ┌──────────────────────────────────┐    │
│    │ Validation (every 250 steps)     │    │
│    ├──────────────────────────────────┤    │
│    │ • model.eval()                   │    │
│    │ • Run 20 validation batches      │    │
│    │ • Compute average loss           │    │
│    │ • Log results                    │    │
│    │ • Save checkpoint (every 5000)   │    │
│    └──────────────────────────────────┘    │
│                                             │
│    ┌──────────────────────────────────┐    │
│    │ HellaSwag Eval (every 250 steps) │    │
│    ├──────────────────────────────────┤    │
│    │ • Test reasoning ability         │    │
│    │ • Multiple choice completion     │    │
│    │ • Log accuracy                   │    │
│    └──────────────────────────────────┘    │
│                                             │
│    ┌──────────────────────────────────┐    │
│    │ Text Generation (every 250 steps)│    │
│    ├──────────────────────────────────┤    │
│    │ • Generate 4 samples             │    │
│    │ • Top-k sampling (k=50)          │    │
│    │ • Monitor text quality           │    │
│    └──────────────────────────────────┘    │
│                                             │
│    ┌──────────────────────────────────┐    │
│    │ Training Step                    │    │
│    ├──────────────────────────────────┤    │
│    │ model.train()                    │    │
│    │ optimizer.zero_grad()            │    │
│    │                                  │    │
│    │ for micro_step in grad_accum:    │    │
│    │   x, y = get_batch()             │    │
│    │   logits, loss = model(x, y)     │    │
│    │   loss.backward()                │    │
│    │                                  │    │
│    │ clip_grad_norm_(1.0)             │    │
│    │ lr = get_lr(step)  # Cosine     │    │
│    │ optimizer.step()                 │    │
│    │                                  │    │
│    │ Log: loss, lr, time, tok/sec     │    │
│    └──────────────────────────────────┘    │
│                                             │
└─────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────┐
│ Cleanup                                     │
├─────────────────────────────────────────────┤
│ • Save final checkpoint                     │
│ • Destroy process group (if DDP)            │
└─────────────────────────────────────────────┘
```

### Gradient Accumulation

Why we need it: GPU memory limits batch size

```
Goal: Effective batch size = 524,288 tokens
GPU can fit: 64 sequences × 1024 tokens = 65,536 tokens

Solution: Accumulate gradients over multiple mini-batches

grad_accum_steps = 524,288 / (64 × 1024 × num_gpus)
                 = 8 steps (for 1 GPU)

Training step:
  1. zero_grad()
  2. for i in range(8):
       x, y = next_batch()
       loss = model(x, y) / 8  # Scale loss
       loss.backward()         # Gradients accumulate!
  3. optimizer.step()           # Update once with accumulated grads
```

### Learning Rate Schedule

```
Cosine Annealing with Warmup

Learning Rate
    │
max_lr ┤     ╱────╲
    │    ╱      ╲
    │   ╱        ╲___
    │  ╱             ╲___
min_lr ┤─╱                  ───────
    │
    └─────────────────────────────────> Steps
      │  │                    │
      0  warmup_steps      max_steps
         (715)             (19073)

Phase 1 (Warmup): Linear increase from 0 → max_lr
  lr = max_lr * (step + 1) / warmup_steps

Phase 2 (Cosine Decay): Smooth decrease max_lr → min_lr
  decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
  coeff = 0.5 * (1 + cos(π * decay_ratio))
  lr = min_lr + coeff * (max_lr - min_lr)

Phase 3 (Post-training): Stay at min_lr
  lr = min_lr
```

### Mixed Precision Training

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits, loss = model(x, y)
```

**Benefits:**
- 2× faster training (FP16/BF16 vs FP32)
- 2× less memory (can use larger batches)
- Minimal accuracy loss

**BFloat16 vs Float16:**
- BFloat16: Same range as FP32, less precision (better for LLMs)
- Float16: Smaller range, requires loss scaling

### Distributed Data Parallel (DDP)

Launch with: `torchrun --standalone --nproc_per_node=8 train.py`

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   GPU 0      │  │   GPU 1      │  │   GPU 2      │
│              │  │              │  │              │
│  Model Copy  │  │  Model Copy  │  │  Model Copy  │
│  (sync'd)    │  │  (sync'd)    │  │  (sync'd)    │
└──────────────┘  └──────────────┘  └──────────────┘
       │                 │                 │
       ├─────────────────┼─────────────────┤
       │    Different Data (no overlap)    │
       └───────────────────────────────────┘
                        │
                   Forward Pass
                        │
                   Backward Pass
                        │
             ┌──────────┴──────────┐
             │  All-Reduce Grads   │  ← Synchronize
             │  (average across    │
             │   all GPUs)         │
             └──────────┬──────────┘
                        │
                  Optimizer Step
                  (all GPUs update
                   identically)
```

**Key points:**
- Each GPU has identical model
- Different data per GPU (distributed sampling)
- Gradients averaged across all GPUs
- All models stay synchronized

---

## Parameter Counts

### GPT-2 Small (124M parameters)

**Config:** n_layer=12, n_head=12, n_embd=768

```
Component                   Shape              Parameters
─────────────────────────────────────────────────────────
Token Embeddings (wte)      (50257, 768)      38,597,376
Position Embeddings (wpe)   (1024, 768)          786,432
                                              ───────────
Embeddings Total                              39,383,808

Per Transformer Block:
  LayerNorm 1               (768,) × 2             1,536
  Attention c_attn          (768, 2304)        1,769,472
  Attention c_proj          (768, 768)           589,824
  LayerNorm 2               (768,) × 2             1,536
  MLP c_fc                  (768, 3072)        2,359,296
  MLP c_proj                (3072, 768)        2,359,296
                                              ───────────
  Per Block Total                              7,080,960

× 12 blocks                                   85,011,520

Final LayerNorm             (768,) × 2             1,536

LM Head (tied with wte)     (768, 50257)              0*
                                              ───────────

Total Parameters                             124,396,864
Non-embedding Parameters                      84,999,424

*Weight tying: LM head shares weights with token embeddings
```

### Scaling to Other Sizes

| Model | n_layer | n_embd | Params/Block | Total Params |
|-------|---------|--------|--------------|--------------|
| **Small** | 12 | 768 | 7.1M | 124M |
| **Medium** | 24 | 1024 | 12.6M | 350M |
| **Large** | 36 | 1280 | 19.7M | 774M |
| **XL** | 48 | 1600 | 30.7M | 1558M |

**Scaling pattern:**
- Doubling depth (~2× parameters)
- Increasing width (quadratic growth in params/block)

---

## Data Flow Visualization

### Complete End-to-End Flow

```
INPUT: "The cat sat on the"
  Tokenized: [464, 3797, 3332, 319, 262]
       ↓
┌─────────────────────────────────────────────────────────┐
│ EMBEDDINGS                                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Token IDs: [464, 3797, 3332, 319, 262]                 │
│       ↓                                                 │
│  Token Embeddings: (5, 768)                             │
│    464   → [0.023, -0.145, ..., 0.891]                  │
│    3797  → [0.512, -0.023, ..., -0.234]                 │
│    ...                                                  │
│       +                                                 │
│  Position Embeddings: (5, 768)                          │
│    pos 0 → [0.012, 0.234, ..., -0.123]                  │
│    pos 1 → [-0.456, 0.789, ..., 0.456]                  │
│    ...                                                  │
│       ↓                                                 │
│  Combined + Dropout: (5, 768)                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────┐
│ BLOCK 1: Attention + MLP                                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: x₀ (5, 768)                                     │
│       ↓                                                 │
│  ┌──────────────────────────────┐                       │
│  │ Attention Path               │                       │
│  ├──────────────────────────────┤                       │
│  │ LayerNorm(x₀) → (5, 768)     │                       │
│  │      ↓                       │                       │
│  │ Q,K,V projection → (5, 2304) │                       │
│  │      ↓                       │                       │
│  │ Multi-head attention         │                       │
│  │   Each token attends to      │                       │
│  │   previous tokens:           │                       │
│  │                              │                       │
│  │   "The" ← ["The"]            │                       │
│  │   "cat" ← ["The", "cat"]     │                       │
│  │   "sat" ← ["The", "cat",     │                       │
│  │             "sat"]           │                       │
│  │   ...                        │                       │
│  │      ↓                       │                       │
│  │ Concatenate + project        │                       │
│  │      → (5, 768)              │                       │
│  └──────────────────────────────┘                       │
│       ↓                                                 │
│  x₁ = x₀ + attn_out  (residual)                         │
│       ↓                                                 │
│  ┌──────────────────────────────┐                       │
│  │ MLP Path                     │                       │
│  ├──────────────────────────────┤                       │
│  │ LayerNorm(x₁) → (5, 768)     │                       │
│  │      ↓                       │                       │
│  │ Expand → (5, 3072)           │                       │
│  │      ↓                       │                       │
│  │ GELU                         │                       │
│  │      ↓                       │                       │
│  │ Project → (5, 768)           │                       │
│  └──────────────────────────────┘                       │
│       ↓                                                 │
│  x₂ = x₁ + mlp_out  (residual)                          │
│                                                         │
│  Output: (5, 768)                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────┐
│ BLOCKS 2-12 (same structure)                            │
│                                                         │
│  Each block refines the representation                  │
│  Information flows through residual stream              │
│                                                         │
└─────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────┐
│ FINAL LAYER NORM                                        │
│                                                         │
│  x_final = LayerNorm(x₁₂)  (5, 768)                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────┐
│ LANGUAGE MODEL HEAD                                     │
│                                                         │
│  Logits = Linear(x_final, vocab_size)                   │
│         = (5, 768) @ (768, 50257)ᵀ                      │
│         = (5, 50257)                                    │
│                                                         │
│  For each position, scores for all possible next tokens:│
│                                                         │
│  Position 0 ("The"):    [... 0.2 ... 8.5 ... -2.1 ...]  │
│  Position 1 ("cat"):    [... 4.1 ... 1.2 ... -0.5 ...]  │
│  Position 2 ("sat"):    [... -1.2 ... 7.8 ... 3.4 ...]  │
│  Position 3 ("on"):     [... 2.5 ... 9.1 ... 1.1 ...]   │
│  Position 4 ("the"):    [... 6.7 ... 2.3 ... 8.9 ...]   │
│                           ↑           ↑          ↑       │
│                        token_1    token_2   token_N      │
│                                                         │
└─────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────┐
│ PREDICTION                                              │
│                                                         │
│  Apply softmax to position 4 ("the"):                   │
│    Highest score: token 13625 = "mat"                   │
│                                                         │
│  Generated text: "The cat sat on the mat"               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Tensor Shape Journey

```
Step                          Shape              Notes
──────────────────────────────────────────────────────────
Input token IDs               (B, T)             Integers
Token embeddings              (B, T, 768)        Lookup
Position embeddings           (T, 768)           Broadcast
Combined embeddings           (B, T, 768)        Addition

Through each block:
  LayerNorm input             (B, T, 768)
  Attention Q,K,V             (B, T, 2304)       Concat
  Q,K,V split                 3 × (B, T, 768)
  Q,K,V reshaped              3 × (B, 12, T, 64) Multi-head
  Attention output            (B, 12, T, 64)
  Concatenated                (B, T, 768)
  After residual              (B, T, 768)        x + attn

  LayerNorm input             (B, T, 768)
  MLP expand                  (B, T, 3072)       4× width
  MLP project                 (B, T, 768)        Back to 768
  After residual              (B, T, 768)        x + mlp

After all blocks              (B, T, 768)
Final LayerNorm               (B, T, 768)
LM Head projection            (B, T, 50257)      Vocab size
Logits                        (B, T, 50257)      Predictions
```

**Key invariant:** The hidden dimension (768) is preserved throughout the network via residual connections!

---

## Key Takeaways

### 1. Modular Design
- **config.py**: All hyperparameters in one place
- **attention.py**: Self-attention mechanism
- **block.py**: Transformer block (attention + MLP)
- **model.py**: Full GPT model assembly
- **data.py**: Efficient data loading
- **train.py**: Training orchestration

### 2. Critical Dimensions
- **n_embd (768)**: Width of the model - main information highway
- **n_layer (12)**: Depth of the model - number of transformer blocks
- **n_head (12)**: Parallelism in attention
- **block_size (1024)**: Maximum context length
- **vocab_size (50257)**: Number of unique tokens

### 3. Modern Features
- ✅ Flash Attention (`F.scaled_dot_product_attention`)
- ✅ Full dropout support (attention, residual, embedding)
- ✅ Configurable bias terms
- ✅ Pre-norm architecture (more stable)
- ✅ Weight tying (embeddings = LM head)
- ✅ Scaled initialization for deep networks
- ✅ Distributed training support (DDP)
- ✅ Mixed precision training (BFloat16)
- ✅ Gradient accumulation

### 4. Training Pipeline
- **Data**: Pre-tokenized shards, infinite iteration
- **Optimization**: AdamW with weight decay
- **Schedule**: Cosine annealing with linear warmup
- **Regularization**: Dropout + gradient clipping
- **Evaluation**: Validation loss, HellaSwag, text generation
- **Checkpointing**: Every 5000 steps

### 5. Information Flow
```
Tokens → Embeddings → [Attention → MLP] × 12 → LayerNorm → Logits
                       ↑____________↑
                       Residual Stream
                       (preserves n_embd dimension)
```

---

## Further Reading

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

### Resources
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

---

*Generated for the modular GPT-2 implementation in this repository*
