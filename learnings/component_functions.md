# What Each Component Actually Does

A functional guide to understanding **why** each piece of the GPT-2 architecture exists and what role it plays.

---

## Table of Contents

1. [Overview: The Big Picture](#overview-the-big-picture)
2. [Token Embeddings: Converting Words to Numbers](#token-embeddings)
3. [Position Embeddings: Teaching Position Awareness](#position-embeddings)
4. [Layer Normalization: Keeping Values Stable](#layer-normalization)
5. [Multi-Head Attention: Gathering Context](#multi-head-attention)
6. [MLP (Feed-Forward): Processing Information](#mlp-feed-forward)
7. [Residual Connections: The Information Highway](#residual-connections)
8. [Dropout: Preventing Memorization](#dropout)
9. [Language Model Head: Making Predictions](#language-model-head)
10. [Complete Example Walkthrough](#complete-example-walkthrough)

---

## Overview: The Big Picture

### What is GPT-2 trying to do?

**Goal:** Given a sequence of words, predict the next word.

```
Input:  "The cat sat on the"
Output: "mat" (predicted next word)
```

### How does it achieve this?

The architecture has **three main jobs**:

1. **Understand each word** (embeddings)
2. **Understand relationships between words** (attention + MLP blocks)
3. **Make predictions** (language model head)

Let's see how each component contributes to these jobs.

---

## Token Embeddings

**File:** `gpt2/model.py` line 50

```python
self.transformer.wte = nn.Embedding(vocab_size, n_embd)
```

### What does it do?

**Converts discrete word IDs into continuous vectors that capture meaning.**

### The Problem It Solves

```
Computers can't understand words directly:
  "cat" → ??? (computer doesn't know what this means)

Need to convert to numbers:
  "cat" → [0.234, -0.891, 0.445, ..., 0.123]  (768 numbers)
```

### How It Works

Think of it as a **lookup table** that maps each word to a unique vector:

```
Vocabulary (50,257 words):
  Word ID    →    768-dimensional vector

  0 ("the")  →   [0.023, -0.145, 0.891, ...]
  1 ("a")    →   [0.512, -0.023, -0.234, ...]
  ...
  3797 ("cat") → [0.234, -0.891, 0.445, ...]
  ...
  50256      →   [-0.123, 0.456, 0.789, ...]
```

### What The Numbers Mean

The 768 numbers represent **semantic features**:

```
Hypothetical dimensions (simplified):
  Dimension 0:   Is it an animal? (0.8 = yes)
  Dimension 1:   Is it concrete? (0.9 = yes)
  Dimension 2:   Size (-0.3 = small)
  Dimension 3:   Can it move? (0.7 = yes)
  ...
  Dimension 767: (complex abstract feature)

For "cat":  [0.8, 0.9, -0.3, 0.7, ..., ???]
For "dog":  [0.8, 0.9, -0.2, 0.7, ..., ???]  ← Similar!
For "the":  [0.0, 0.1, 0.0, 0.0, ..., ???]  ← Very different!
```

**Key insight:** Similar words get similar vectors!

### Why 768 Dimensions?

```
Too few dimensions (e.g., 10):
  Can't capture enough nuance
  "cat" and "dog" might be identical

Too many dimensions (e.g., 10,000):
  Wasteful, hard to train
  Most dimensions would be redundant

768 dimensions:
  Sweet spot for GPT-2 Small
  Enough to capture rich semantics
  Computationally efficient
```

### Concrete Example

```python
Input sentence: "The cat sat"
Token IDs:      [464, 3797, 3332]

Embedding lookup:
  464  ("The") → [0.023, -0.145, ..., 0.891]  (768 numbers)
  3797 ("cat") → [0.234, -0.891, ..., 0.445]  (768 numbers)
  3332 ("sat") → [-0.123, 0.456, ..., 0.789] (768 numbers)

Output shape: (3, 768)  # 3 words, each with 768 features
```

### What Makes Good Embeddings?

The model **learns** these embeddings during training:

```
Initially (random):
  "cat" → [0.01, 0.02, -0.03, ...]  (random noise)
  "dog" → [-0.05, 0.01, 0.04, ...]  (also random, unrelated)

After training:
  "cat" → [0.234, -0.891, 0.445, ...]
  "dog" → [0.221, -0.875, 0.453, ...]  ← Learned to be similar!

Why? The model learned that "cat" and "dog" appear in similar contexts:
  "The ___ is sleeping"
  "I pet the ___"
  "The ___ is cute"
```

---

## Position Embeddings

**File:** `gpt2/model.py` line 55

```python
self.transformer.wpe = nn.Embedding(block_size, n_embd)
```

### What does it do?

**Tells the model WHERE each word is in the sentence.**

### The Problem It Solves

Without position information, the model can't tell:

```
"The cat chased the dog"
vs
"The dog chased the cat"

Without positions:
  Both sentences have the same words!
  Model would give the same answer for both
  But they mean different things!
```

### How It Works

Each **position** gets its own learned vector:

```
Position    →    768-dimensional vector

Position 0  →   [0.012, 0.234, -0.123, ...]  (first word)
Position 1  →   [-0.456, 0.789, 0.456, ...]  (second word)
Position 2  →   [0.234, -0.123, 0.567, ...]  (third word)
...
Position 1023 → [-0.789, 0.234, -0.456, ...] (last position)
```

### Combining Token + Position

```
Sentence: "The cat sat"

Step 1: Get token embeddings
  "The" (pos 0): [0.023, -0.145, ..., 0.891]
  "cat" (pos 1): [0.234, -0.891, ..., 0.445]
  "sat" (pos 2): [-0.123, 0.456, ..., 0.789]

Step 2: Get position embeddings
  Position 0: [0.012, 0.234, ..., -0.123]
  Position 1: [-0.456, 0.789, ..., 0.456]
  Position 2: [0.234, -0.123, ..., 0.567]

Step 3: Add them together (element-wise)
  "The" at pos 0: [0.023, -0.145, ...] + [0.012, 0.234, ...]
                = [0.035, 0.089, ..., 0.768]

  "cat" at pos 1: [0.234, -0.891, ...] + [-0.456, 0.789, ...]
                = [-0.222, -0.102, ..., 0.901]

  "sat" at pos 2: [-0.123, 0.456, ...] + [0.234, -0.123, ...]
                = [0.111, 0.333, ..., 1.356]

Now each word has BOTH meaning AND position information!
```

### Why This Matters

```
"I ate the pizza"  vs  "The pizza ate me"
   ↑              ↑       ↑

Position 0:  "I"       vs  "The"
Position 1:  "ate"     vs  "pizza"
Position 2:  "the"     vs  "ate"
Position 3:  "pizza"   vs  "me"

Even though they use the same words, positions make the meaning clear!
```

### Learned vs Fixed Positions

**GPT-2 uses learned positions** (not sinusoidal like original Transformer):

```
Learned (GPT-2):
  ✅ Model discovers optimal position representation
  ✅ Can encode more than just "distance"
  ❌ Fixed maximum length (1024 tokens)

Sinusoidal (Original Transformer):
  ✅ Can handle any length (mathematical formula)
  ❌ Less flexible (formula is fixed)
```

---

## Layer Normalization

**File:** `gpt2/block.py` lines 89, 96

```python
self.ln_1 = nn.LayerNorm(n_embd)  # Before attention
self.ln_2 = nn.LayerNorm(n_embd)  # Before MLP
```

### What does it do?

**Keeps the numbers in a reasonable range so training doesn't explode.**

### The Problem It Solves

```
Without normalization, values can grow uncontrollably:

Layer 1 output: [0.5, -0.3, 0.8, ...]       (reasonable)
Layer 2 output: [5.2, -3.1, 8.4, ...]       (getting larger)
Layer 3 output: [52.3, -31.2, 84.5, ...]    (too large!)
Layer 4 output: [523.1, -312.4, 845.2, ...] (EXPLOSION! 💥)

Gradients become infinity → training fails
```

### How It Works

**Normalizes each layer's output to have mean=0 and variance=1:**

```python
Input:  [10.0, 20.0, 30.0]  # Mean=20, Std=8.16

Step 1: Subtract mean
  [10-20, 20-20, 30-20] = [-10.0, 0.0, 10.0]  # Mean=0

Step 2: Divide by standard deviation
  [-10/8.16, 0/8.16, 10/8.16] = [-1.22, 0.0, 1.22]  # Std=1

Step 3: Scale and shift (learned parameters γ and β)
  γ * [-1.22, 0.0, 1.22] + β

Output: Normalized values (mean≈0, std≈1)
```

### Why This Helps

```
After normalization:

Layer 1 output: [-0.5, 0.3, -0.8, ...]  (normalized)
Layer 2 output: [0.2, -0.1, 0.6, ...]   (still normalized)
Layer 3 output: [-0.3, 0.5, -0.2, ...]  (still normalized)
...
Layer 12 output: [0.1, -0.4, 0.7, ...]  (still normalized)

Values stay in a reasonable range → stable training ✅
```

### Pre-Norm vs Post-Norm

**Our implementation uses Pre-Norm** (normalize BEFORE the operation):

```python
# Pre-Norm (GPT-2, our implementation)
x = x + attention(LayerNorm(x))
      ↑          ↑
      |          └─ Normalize BEFORE attention
      └─ Then add to residual

# Post-Norm (original Transformer)
x = LayerNorm(x + attention(x))
    ↑         └─ Add THEN normalize
    └─ Normalization happens AFTER

Why Pre-Norm is better:
  ✅ Gradients flow directly through residual path
  ✅ More stable for deep networks
  ✅ Easier to train
```

### Concrete Example

```python
Input to LayerNorm: [5.2, -3.1, 8.4, 2.5, -1.2, 6.8, ...]  (768 values)

Mean = sum / 768 = 2.1
Std = sqrt(variance) = 4.3

Normalized:
  (5.2 - 2.1) / 4.3 = 0.72
  (-3.1 - 2.1) / 4.3 = -1.21
  (8.4 - 2.1) / 4.3 = 1.47
  ...

Output: [0.72, -1.21, 1.47, ...]  (mean≈0, std≈1)
```

### Learnable Parameters

LayerNorm has two **learned** parameters:

```python
γ (gamma): Scale parameter (768 values)
β (beta):  Shift parameter (768 values)

Final output: γ * normalized + β

This allows the model to learn:
  "Maybe I want dimension 5 to have mean=0.5"
  "Maybe I want dimension 12 to have std=2.0"
```

---

## Multi-Head Attention

**File:** `gpt2/attention.py`

### What does it do?

**Allows each word to "look at" and gather information from previous words.**

### The Problem It Solves

```
Understanding context requires looking at other words:

"The cat sat on the mat because it was tired"
                              ↑
                              What does "it" refer to?

Need to look back at "cat" to understand!
```

### The Core Idea

Each word can **attend to** (focus on) other words:

```
Sentence: "The cat sat on the mat"

When processing "mat":
  How much should I focus on "The"?   → 0.05 (low attention)
  How much should I focus on "cat"?   → 0.10 (medium)
  How much should I focus on "sat"?   → 0.15 (medium)
  How much should I focus on "on"?    → 0.25 (high)
  How much should I focus on "the"?   → 0.40 (high)
  How much should I focus on "mat"?   → 0.05 (low - itself)
                                        ─────
                                        1.00 (probabilities sum to 1)

These weights tell us what to pay attention to!
```

### How Attention Works (Simplified)

**Step 1: Create Queries, Keys, and Values**

Think of it like a **search system**:

```
Query (Q):  "What am I looking for?"
Key (K):    "What do I offer?"
Value (V):  "What information do I have?"

Example: Processing "mat"

"mat"'s Query:  "I need to find words related to surfaces/objects"

Each word's Key:
  "The":  "I'm a determiner"          (doesn't match query well)
  "cat":  "I'm an animal"             (doesn't match query well)
  "sat":  "I'm an action"             (matches a bit)
  "on":   "I'm a positional word"     (matches well!)
  "the":  "I'm a determiner"          (doesn't match query well)
  "mat":  "I'm a surface object"      (matches perfectly!)

Attention = How well Query matches each Key
```

**Step 2: Compute Attention Scores**

```
Score(Q, K) = Q · K / √d
              ↑   ↑   ↑
              |   |   └─ Scale factor (prevents large values)
              |   └─ Dot product (measures similarity)
              └─ Query vector

Example scores for "mat" attending to each word:
  "mat" → "The": 0.2   (low similarity)
  "mat" → "cat": 0.5   (medium)
  "mat" → "sat": 0.8   (higher)
  "mat" → "on":  2.1   (high!)
  "mat" → "the": 1.5   (high)
  "mat" → "mat": 1.0   (medium - itself)
```

**Step 3: Convert to Probabilities (Softmax)**

```
Scores: [0.2, 0.5, 0.8, 2.1, 1.5, 1.0]

After softmax:
  Attention weights: [0.05, 0.10, 0.15, 0.40, 0.25, 0.05]
                      ↑     ↑     ↑     ↑     ↑     ↑
                      The   cat   sat   on    the   mat

Sum = 1.00 (probabilities!)
```

**Step 4: Weighted Sum of Values**

```
Each word's Value vector contains its "information":
  "The":  [0.1, -0.2, 0.3, ...]
  "cat":  [0.5, 0.8, -0.1, ...]
  "sat":  [-0.3, 0.4, 0.7, ...]
  "on":   [0.9, -0.5, 0.2, ...]
  "the":  [0.1, -0.2, 0.3, ...]
  "mat":  [0.6, 0.3, -0.4, ...]

Weighted sum (what "mat" learns):
  0.05 * [0.1, -0.2, 0.3, ...]    (5% from "The")
+ 0.10 * [0.5, 0.8, -0.1, ...]    (10% from "cat")
+ 0.15 * [-0.3, 0.4, 0.7, ...]    (15% from "sat")
+ 0.40 * [0.9, -0.5, 0.2, ...]    (40% from "on")  ← Most weight!
+ 0.25 * [0.1, -0.2, 0.3, ...]    (25% from "the")
+ 0.05 * [0.6, 0.3, -0.4, ...]    (5% from "mat")
= [0.52, -0.18, 0.25, ...]        (combined information)

"mat" now has information from all previous words,
weighted by relevance!
```

### Why "Multi-Head" Attention?

**Different heads can focus on different types of relationships:**

```
Same sentence: "The cat sat on the mat because it was tired"

Head 1 (Positional patterns):
  "mat" → "the" (adjacent words)
  "tired" → "was" (adjacent words)

Head 2 (Syntactic patterns):
  "sat" → "cat" (subject-verb agreement)
  "was" → "it" (auxiliary-subject)

Head 3 (Semantic patterns):
  "it" → "cat" (pronoun resolution)
  "mat" → "on" (spatial relationship)

Head 4 (Long-range):
  "tired" → "cat" (who is tired?)

...12 heads total, each specializing in different patterns
```

### Causal Masking (Autoregressive)

**Critical:** Each word can only look at **previous** words, not future ones!

```
Sentence: "The cat sat on the"

When predicting next word after "the":

✅ CAN look at:     "The", "cat", "sat", "on", "the"
❌ CANNOT look at:  Future words (don't exist yet!)

Attention mask:
         The  cat  sat  on   the
    The  ✓    ✗    ✗    ✗    ✗
    cat  ✓    ✓    ✗    ✗    ✗
    sat  ✓    ✓    ✓    ✗    ✗
    on   ✓    ✓    ✓    ✓    ✗
    the  ✓    ✓    ✓    ✓    ✓

This is called "causal" or "autoregressive" attention
```

### Concrete Example

```python
Input:  "The cat sat"
Shape:  (3, 768)  # 3 words, 768 dimensions each

After attention:
  "The": Focused only on itself [0.023, -0.145, ...]
  "cat": Combined "The" + "cat" [0.156, -0.234, ...]
  "sat": Combined "The" + "cat" + "sat" [0.089, 0.123, ...]

Output: (3, 768)  # Same shape, but enriched with context!
```

---

## MLP (Feed-Forward)

**File:** `gpt2/block.py` lines 17-66

### What does it do?

**Processes each word's representation independently to extract complex patterns.**

### The Problem It Solves

Attention gathers information, but doesn't **transform** it:

```
After attention, "cat" has information from context:
  [0.156, -0.234, 0.567, ...]

But we need to:
  ✅ Extract higher-level features
  ✅ Apply non-linear transformations
  ✅ Recognize complex patterns

Example: "The cat sat" → Recognize this is a complete clause
```

### How It Works

**Two-layer neural network with expansion:**

```
Input:  (768 dimensions)
   ↓
Expand to 4× size:  (3072 dimensions)
   ↓
Apply non-linear activation (GELU)
   ↓
Compress back:  (768 dimensions)
   ↓
Output: Same size as input, but transformed
```

### Step-by-Step

```python
Step 1: Expand (Linear layer 768 → 3072)
  Input:  [0.5, -0.3, 0.8, ..., 0.2]  (768 numbers)
  Weight: (768, 3072) matrix
  Output: [1.2, -0.5, ..., 0.9]       (3072 numbers)

Step 2: Apply GELU activation
  GELU is a smooth non-linearity:
  - Small positive → keeps positive
  - Small negative → makes more negative
  - Large values → keeps mostly unchanged

  [1.2, -0.5, 0.3, ..., 0.9]
    ↓ GELU
  [1.19, -0.15, 0.28, ..., 0.89]

Step 3: Compress (Linear layer 3072 → 768)
  Input:  [1.19, -0.15, 0.28, ..., 0.89]  (3072 numbers)
  Weight: (3072, 768) matrix
  Output: [0.7, -0.1, 0.4, ..., 0.3]      (768 numbers)
```

### Why 4× Expansion?

```
More dimensions = more capacity to learn complex functions

Analogy:
  Input (768 dim):    Simple representation
  Expanded (3072):    Rich space to compute complex patterns
  Output (768 dim):   Compressed insights

Like:
  Question (concise) → Think deeply (expand) → Answer (concise)
```

### What Does MLP Learn?

**Pattern detection and feature extraction:**

```
Example patterns the MLP might learn:

Neuron 1:  Detects "is this a noun phrase?"
  Input: [cat representations] → Output: 0.9 (yes, high score)

Neuron 2:  Detects "is this a verb?"
  Input: [cat representations] → Output: 0.1 (no, low score)

Neuron 3:  Detects "is this animate?"
  Input: [cat representations] → Output: 0.8 (yes, animals are animate)

...3072 neurons in the hidden layer, each detecting different patterns
```

### MLP vs Attention: What's the Difference?

```
Attention:
  ✅ Gathers information from other words
  ✅ "Communication" between tokens
  ❌ Doesn't transform much (mostly linear)

  Example: "cat" learns about "The" and "sat"

MLP:
  ✅ Processes each word independently
  ✅ "Computation" on individual tokens
  ✅ Non-linear transformations
  ❌ Doesn't look at other words

  Example: "cat" representation becomes more refined
```

### Working Together

```
Transformer block = Attention + MLP

Step 1: Attention
  Gather context: "cat" ← learns from "The"

Step 2: MLP
  Process: "cat with context from The" → extract features

Result: Rich representation that has both context AND processing
```

---

## Residual Connections

**File:** `gpt2/block.py` lines 120, 124

```python
x = x + self.attn(self.ln_1(x))  # Residual connection
x = x + self.mlp(self.ln_2(x))   # Residual connection
```

### What does it do?

**Creates a "highway" that allows information to skip layers.**

### The Problem It Solves

**Without residuals, deep networks suffer from vanishing gradients:**

```
12-layer network without residuals:

Input → Layer 1 → Layer 2 → ... → Layer 12 → Output
                                        ↓
                                   Gradient: 1.0

Backpropagation:
Layer 11 ← gradient: 0.8
Layer 10 ← gradient: 0.6
Layer 9  ← gradient: 0.4
Layer 8  ← gradient: 0.2
Layer 7  ← gradient: 0.1
Layer 6  ← gradient: 0.05
...
Layer 1  ← gradient: 0.001  ← Almost zero! Doesn't learn!

Problem: Early layers don't learn because gradients "vanish"
```

### How It Works

**Add the input back to the output:**

```python
# Without residual:
output = layer(input)

# With residual:
output = input + layer(input)
         ↑      ↑
         |      └─ Transformed by layer
         └─ Original input (unchanged path)
```

### Visual Representation

```
Residual connection creates two paths:

Path 1 (Residual - Direct):
  Input ─────────────────────────────→ +
    ↓                                   ↓
    └──────────────────────────────→ Output

Path 2 (Transformation):
  Input → LayerNorm → Attention → +
                                   ↓
                                Output

The "+" combines both paths:
  output = input + attention(norm(input))
```

### Why This Helps

**Gradient flow:**

```
With residuals:

∂output/∂input = 1 + ∂attention/∂input
                 ↑   └─ Can be small
                 └─ Always 1! Gradient can always flow!

Backpropagation:
Layer 12 ← gradient: 1.0
Layer 11 ← gradient: 0.95  (mostly 1.0 from residual)
Layer 10 ← gradient: 0.92
...
Layer 1  ← gradient: 0.65  ← Still strong! Can learn!
```

### The Residual Stream Analogy

Think of it as a **river**:

```
          ┌─ Attention ─┐
          ↓             ↓
River ────────────────────────→ (main flow)
     ↓                 ↓
     └── MLP ──┘      ↓
                      ↓
     ─────────────────────────→ (continues)

- Main river (residual stream) flows unchanged
- Tributaries (attention, MLP) add information
- River never stops flowing!
```

### Concrete Example

```python
# At some layer:
x = [0.5, -0.3, 0.8, ...]  (768 values)

attention_output = [0.1, 0.05, -0.2, ...]  (768 values)

# Without residual:
x_new = attention_output  # Throw away original!
      = [0.1, 0.05, -0.2, ...]

# With residual:
x_new = x + attention_output  # Keep original + add new info
      = [0.5, -0.3, 0.8, ...] + [0.1, 0.05, -0.2, ...]
      = [0.6, -0.25, 0.6, ...]  # Combined!

Information from both paths preserved!
```

### Multiple Residual Connections

```
GPT-2 has 24 residual additions (2 per block × 12 blocks):

Input embeddings
  ↓
+ Attention output (block 1)
  ↓
+ MLP output (block 1)
  ↓
+ Attention output (block 2)
  ↓
+ MLP output (block 2)
  ↓
...
  ↓
+ Attention output (block 12)
  ↓
+ MLP output (block 12)
  ↓
Final representation

Each addition refines the representation while preserving core information
```

---

## Dropout

**File:** Various locations

```python
self.dropout = nn.Dropout(config.dropout)  # e.g., 0.1 = 10%
```

### What does it do?

**Randomly "turns off" some neurons during training to prevent over-reliance on specific patterns.**

### The Problem It Solves

**Overfitting: Memorizing training data instead of learning patterns:**

```
Without dropout:

Training: Model sees "The cat sat on the mat" 1000 times
  → Learns to predict "mat" after "the" perfectly (100% accuracy)

Testing: Model sees "The cat sat on the rug"
  → Predicts "mat" (wrong! Should predict "rug")
  → Has memorized specific examples, not learned general patterns

With dropout:

Training: Model can't rely on specific neurons (they're randomly off)
  → Must learn robust patterns that work with different subsets

Testing: All neurons active
  → Better generalization!
```

### How It Works

**During training, randomly set activations to zero:**

```python
# Forward pass with dropout=0.1 (10% drop rate)

Original activations:
  [0.5, -0.3, 0.8, 0.2, -0.1, 0.6, 0.4, -0.2, 0.9, 0.1]

Random mask (90% keep, 10% drop):
  [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]  ← 0 means "turn off"

After dropout:
  [0.5, -0.3, 0.0, 0.2, -0.1, 0.6, 0.0, -0.2, 0.9, 0.1]
       ↑          ↑                        ↑
       |          └─ Dropped!              └─ Dropped!
       └─ Kept

Scaled by 1/(1-dropout) = 1/0.9 = 1.11 to maintain expected value:
  [0.56, -0.33, 0.0, 0.22, -0.11, 0.67, 0.0, -0.22, 1.0, 0.11]
```

### Why Scaling?

```
Without scaling:
  Expected activation with dropout: 0.9 × original
  Expected activation without dropout: 1.0 × original
  → Different behavior between training and inference!

With scaling (× 1.11):
  Expected activation with dropout: 0.9 × 1.11 × original = 1.0 × original
  Expected activation without dropout: 1.0 × original
  → Same expected value! ✅
```

### Where Dropout is Applied

**In our implementation (4 places):**

```
1. Embedding Dropout:
   Token + Position embeddings → Dropout → First block

2. Attention Dropout:
   In scaled_dot_product_attention (on attention weights)

3. Residual Dropout (after attention projection):
   Attention output → Dropout → Add to residual

4. MLP Dropout:
   MLP output → Dropout → Add to residual
```

### Effect of Dropout Rate

```
dropout=0.0 (no dropout):
  ✅ Fastest training
  ❌ Overfits (memorizes training data)

dropout=0.1 (light):
  ✅ Good balance
  ✅ Prevents overfitting
  ⚠️ Slightly slower convergence

dropout=0.3 (heavy):
  ✅ Very robust (doesn't overfit)
  ❌ Slow convergence
  ❌ May underfit (can't learn complex patterns)
```

### During Inference (Evaluation)

```python
# Training mode:
model.train()
x = dropout(x)  # Randomly drops 10%

# Evaluation mode:
model.eval()
x = dropout(x)  # Does nothing (all neurons active)
```

### Concrete Example

```
Layer output (10 dimensions):
  [0.5, -0.3, 0.8, 0.2, -0.1, 0.6, 0.4, -0.2, 0.9, 0.1]

Training forward pass 1:
  Random mask: [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
  Output: [0.56, -0.33, 0.0, 0.22, 0.0, 0.67, 0.44, -0.22, 0.0, 0.11]

Training forward pass 2 (different mask!):
  Random mask: [1, 0, 1, 1, 1, 0, 1, 1, 1, 0]
  Output: [0.56, 0.0, 0.89, 0.22, -0.11, 0.0, 0.44, -0.22, 1.0, 0.0]

This forces the model to work with different subsets of neurons!
```

---

## Language Model Head

**File:** `gpt2/model.py` line 73

```python
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
```

### What does it do?

**Converts the final hidden representation into probabilities for each word in the vocabulary.**

### The Problem It Solves

```
After all transformer blocks, we have:
  Hidden state: [0.234, -0.891, 0.445, ..., 0.123]  (768 numbers)

But we need to predict:
  Next word: "mat" (one of 50,257 possible words)

How do we go from 768 numbers to a word prediction?
```

### How It Works

**Linear projection + softmax:**

```
Step 1: Project to vocabulary size
  Input:  [0.234, -0.891, 0.445, ..., 0.123]  (768 dimensions)
  Weight: (768, 50257) matrix
  Output: [2.3, -1.5, ..., 4.8, ..., 0.9]     (50,257 scores)
                           ↑
                           Score for word "mat"

Step 2: Convert scores to probabilities (softmax)
  Scores: [2.3, -1.5, 0.8, ..., 4.8, ..., 0.9]

  Softmax: e^(score_i) / Σ(e^(score_j))

  Probabilities:
    P("the")   = 0.05   (score: 2.3)
    P("a")     = 0.001  (score: -1.5)
    P("cat")   = 0.02   (score: 0.8)
    ...
    P("mat")   = 0.42   (score: 4.8)  ← Highest!
    ...
    P("rug")   = 0.03   (score: 0.9)

  Sum of all probabilities = 1.0
```

### Concrete Example

```
Input: "The cat sat on the"

Final hidden state for "the": [0.234, -0.891, ..., 0.123]

LM Head projection:
  [0.234, -0.891, ..., 0.123] @ W^T
  = [score_0, score_1, ..., score_50256]

Vocabulary scores:
  Word 0 ("!"):         -3.2
  Word 1 ("<|endoftext|>"): -5.1
  ...
  Word 2343 ("mat"):    4.8  ← High score!
  Word 2344 ("rug"):    3.1
  Word 2345 ("floor"):  2.9
  ...
  Word 50256:          -2.1

After softmax:
  P("mat")   = 0.42  ← Most likely
  P("rug")   = 0.08
  P("floor") = 0.06
  ...

Prediction: "mat" (has highest probability)
```

### Why Weight Tying?

**LM head shares weights with token embedding:**

```python
self.transformer.wte.weight = self.lm_head.weight
```

This means:

```
Token Embedding: word_id → vector (50257 → 768)
  "mat" (id 2343) → [0.234, -0.891, ..., 0.123]

LM Head: vector → word_scores (768 → 50257)
  [0.234, -0.891, ..., 0.123] → score for "mat"

These are TRANSPOSES of each other!

Intuition:
  If "mat"'s embedding is [0.234, -0.891, ...]
  And hidden state is similar: [0.220, -0.875, ...]
  Then dot product will be high → high score for "mat"!

Words with similar meanings have similar embeddings
→ Similar hidden states predict similar words
```

### Training vs Inference

**Training: Learn from correct answer**

```
Input: "The cat sat on the"
Target: "mat"

Model predicts:
  P("mat")   = 0.42
  P("rug")   = 0.08
  P("floor") = 0.06
  ...

Loss = -log(P("mat")) = -log(0.42) = 0.87
       ↑
       Cross-entropy loss (want to maximize P("mat"))

Gradient tells model:
  "Increase score for 'mat'"
  "Decrease scores for other words"
```

**Inference: Sample or pick best**

```
Model predicts:
  P("mat")   = 0.42
  P("rug")   = 0.08
  P("floor") = 0.06
  ...

Strategy 1: Greedy (pick highest)
  → "mat"

Strategy 2: Sample from distribution
  → 42% chance "mat", 8% chance "rug", etc.
  → Introduces randomness/creativity

Strategy 3: Top-k sampling (sample from top 50 words)
  → More diverse than greedy
  → Less random than full sampling
```

---

## Complete Example Walkthrough

Let's trace **"The cat sat on the"** through the entire model to predict **"mat"**.

### Input Processing

```
Step 1: Tokenization
  Text: "The cat sat on the"
  Tokens: [464, 3797, 3332, 319, 262]

Step 2: Token Embeddings (vocab_size=50257, n_embd=768)
  464  ("The") → [0.023, -0.145, 0.891, ...]  (768-dim)
  3797 ("cat") → [0.234, -0.891, 0.445, ...]
  3332 ("sat") → [-0.123, 0.456, 0.789, ...]
  319  ("on")  → [0.345, 0.123, -0.234, ...]
  262  ("the") → [0.023, -0.145, 0.891, ...]

Step 3: Position Embeddings
  pos 0 → [0.012, 0.234, -0.123, ...]
  pos 1 → [-0.456, 0.789, 0.456, ...]
  pos 2 → [0.234, -0.123, 0.567, ...]
  pos 3 → [0.567, -0.234, 0.123, ...]
  pos 4 → [-0.123, 0.456, -0.234, ...]

Step 4: Combine (token + position)
  Token 0: [0.023, -0.145, ...] + [0.012, 0.234, ...]
         = [0.035, 0.089, ...]
  Token 1: [0.234, -0.891, ...] + [-0.456, 0.789, ...]
         = [-0.222, -0.102, ...]
  ...

Step 5: Embedding Dropout
  Randomly set 10% to zero (during training)

Initial hidden states: (5, 768)
```

### Block 1 Processing

**Processing token 4 ("the") in detail:**

```
Input to Block 1: x = [-0.100, 0.311, ..., 0.657]  (768-dim)

┌─────────────────────────────────────────────────┐
│ Attention Path                                  │
└─────────────────────────────────────────────────┘

Step 1: LayerNorm
  x_norm = LayerNorm(x)
         = [-0.234, 0.456, ..., 0.789]

Step 2: Multi-Head Attention (12 heads)
  Create Q, K, V for all 5 tokens

  For token 4 ("the"):
    Query: "What should I attend to?"

    Compute attention with all previous tokens:
      Token 0 ("The"): 0.15  (some attention)
      Token 1 ("cat"): 0.20  (more attention)
      Token 2 ("sat"): 0.25  (even more)
      Token 3 ("on"):  0.35  (high attention - related to preposition)
      Token 4 ("the"): 0.05  (low attention - itself)

    Weighted sum of values:
      0.15 * V_0 + 0.20 * V_1 + 0.25 * V_2 + 0.35 * V_3 + 0.05 * V_4
      = [0.345, -0.123, ..., 0.456]

  attn_output = [0.345, -0.123, ..., 0.456]

Step 3: Residual Dropout
  attn_output = Dropout(attn_output)
              = [0.383, -0.137, ..., 0.507]  (scaled up where not dropped)

Step 4: Residual Connection
  x = x + attn_output
    = [-0.100, 0.311, ...] + [0.383, -0.137, ...]
    = [0.283, 0.174, ..., 1.164]

┌─────────────────────────────────────────────────┐
│ MLP Path                                        │
└─────────────────────────────────────────────────┘

Step 5: LayerNorm
  x_norm = LayerNorm(x)
         = [0.156, 0.089, ..., 0.567]

Step 6: MLP (768 → 3072 → 768)
  Expand:   [0.156, 0.089, ...] → (768 → 3072)
          = [1.234, -0.567, ..., 0.890]  (3072-dim)

  GELU:     [1.234, -0.567, ..., 0.890]
          = [1.228, -0.178, ..., 0.884]  (smooth activation)

  Project:  [1.228, -0.178, ...] → (3072 → 768)
          = [0.234, -0.089, ..., 0.345]  (768-dim)

  mlp_output = [0.234, -0.089, ..., 0.345]

Step 7: MLP Dropout
  mlp_output = Dropout(mlp_output)
             = [0.260, -0.099, ..., 0.383]

Step 8: Residual Connection
  x = x + mlp_output
    = [0.283, 0.174, ...] + [0.260, -0.099, ...]
    = [0.543, 0.075, ..., 1.547]

Output from Block 1: [0.543, 0.075, ..., 1.547]
```

### Blocks 2-12

```
Block 2:
  Input:  [0.543, 0.075, ..., 1.547]
  Attention → Residual → MLP → Residual
  Output: [0.689, -0.123, ..., 1.234]

Block 3:
  Input:  [0.689, -0.123, ..., 1.234]
  Attention → Residual → MLP → Residual
  Output: [0.456, 0.234, ..., 0.987]

... (Blocks 4-11)

Block 12:
  Input:  [0.321, -0.456, ..., 1.123]
  Attention → Residual → MLP → Residual
  Output: [0.234, -0.891, ..., 0.445]

Final hidden state for token 4: [0.234, -0.891, ..., 0.445]
```

### Final Prediction

```
Step 1: Final LayerNorm
  x = LayerNorm([0.234, -0.891, ..., 0.445])
    = [0.245, -0.923, ..., 0.456]

Step 2: LM Head (768 → 50257)
  logits = x @ W^T
         = [0.245, -0.923, ...] @ (768, 50257)
         = [-3.2, -5.1, ..., 4.8, ..., 2.1]
            ↑     ↑        ↑         ↑
            !   <|eos|>   mat      rug

Step 3: Softmax
  probabilities = softmax(logits)

  P("the")      = 0.02
  P("a")        = 0.01
  P("mat")      = 0.42  ← Highest!
  P("rug")      = 0.08
  P("floor")    = 0.06
  P("carpet")   = 0.05
  ...
  (sum = 1.0)

Prediction: "mat" (42% probability)

Generated text: "The cat sat on the mat"
```

### What Happened?

```
1. Embeddings: Converted words to vectors with position info

2. Block 1-12: Each block:
   - Attention: "the" learned it follows a preposition ("on")
   - MLP: Recognized pattern "X on the Y" (X sits on surface Y)

3. By Block 12:
   - Model understands: "cat" is subject
   - "sat on" implies a surface
   - "the" precedes a noun (likely the surface)
   - Context suggests a resting surface

4. LM Head:
   - Outputs high score for words like "mat", "rug", "floor"
   - Low score for words like "tree", "sky", "quickly"

5. Prediction:
   - "mat" has highest probability
   - Makes sense: cats sit on mats!
```

---

## Key Takeaways

### Information Flow

```
Raw Text
   ↓
Tokens (discrete IDs)
   ↓
Embeddings (continuous vectors with position)
   ↓
Block 1: Attention (gather context) + MLP (process)
   ↓
Block 2: Attention + MLP (refine)
   ↓
...
   ↓
Block 12: Attention + MLP (final refinement)
   ↓
Final hidden state (rich representation)
   ↓
LM Head (convert to word probabilities)
   ↓
Predicted next word
```

### What Each Component Does

| Component | Job | Analogy |
|-----------|-----|---------|
| **Token Embeddings** | Convert words to numbers | Dictionary: word → meaning vector |
| **Position Embeddings** | Add position information | Page numbers in a book |
| **LayerNorm** | Keep values stable | Thermostat (keeps temperature stable) |
| **Attention** | Gather context | Looking at related words |
| **MLP** | Process information | Thinking about what you read |
| **Residual** | Preserve information | Main highway with exits |
| **Dropout** | Prevent memorization | Practice with random handicaps |
| **LM Head** | Make predictions | Guess the next word |

### The Magic

The model learns ALL of this from examples:

```
Training: Show "The cat sat on the mat" millions of times
  → Model learns patterns
  → Adjusts embeddings, attention, MLP weights
  → Gets better at predicting

Result: Can generate coherent text!
```

---

*This guide explains the **functional role** of each component. For architectural details, see `gpt2_architecture_guide.md`. For why these choices matter, see `ablation_study.md`.*
