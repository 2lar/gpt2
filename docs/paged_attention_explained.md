# Paged Attention: Deep Dive

Complete guide to understanding and using Paged Attention for efficient KV cache management.

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Problem](#the-problem)
3. [The Solution](#the-solution)
4. [How It Works](#how-it-works)
5. [Implementation Details](#implementation-details)
6. [Comparison with Standard KV Cache](#comparison-with-standard-kv-cache)
7. [Usage Examples](#usage-examples)
8. [Performance Analysis](#performance-analysis)
9. [Advanced Topics](#advanced-topics)

---

## Introduction

**Paged Attention** is a memory management technique for KV caches in Large Language Models, introduced in the vLLM paper ([Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)).

It treats KV cache like **virtual memory in operating systems**, using a page table to map logical positions to physical memory blocks.

### Key Innovation

Instead of storing KV cache in contiguous memory per sequence, Paged Attention:
- Divides cache into fixed-size **blocks** (like OS memory pages)
- Uses **block tables** to map logical→physical addresses
- Enables **sharing** and **dynamic allocation**

### Benefits

- ✅ **Near-zero fragmentation** (only last block partially filled)
- ✅ **No over-allocation** (allocate blocks on-demand)
- ✅ **Memory sharing** (for beam search, speculative decoding)
- ✅ **Higher throughput** (fit more sequences in same memory)

---

## The Problem

### Traditional KV Cache

In standard transformer generation, we cache K and V to avoid recomputation:

```python
# Standard KV cache (contiguous memory)
sequences = {
    0: {
        'k_cache': torch.zeros(n_head, max_len, head_dim),  # Pre-allocate
        'v_cache': torch.zeros(n_head, max_len, head_dim),
        'current_len': 0,
    },
    1: {
        'k_cache': torch.zeros(n_head, max_len, head_dim),  # Another allocation
        'v_cache': torch.zeros(n_head, max_len, head_dim),
        'current_len': 0,
    }
}
```

### Problems with Standard Approach

#### 1. **Over-Allocation**

Must pre-allocate `max_len` memory for each sequence:

```
Sequence 1: [K₀, K₁, K₂, _, _, _, _, _, _, _]  ← 70% wasted!
           Used: 3 tokens, Allocated: 10 tokens

Sequence 2: [K₀, K₁, K₂, K₃, K₄, _, _, _, _, _]  ← 50% wasted!
           Used: 5 tokens, Allocated: 10 tokens
```

**Waste**: Average 60% memory wasted on unused pre-allocation.

#### 2. **External Fragmentation**

When a sequence finishes, its memory can't be reused by others:

```
Initial:
  Seq 0: [####________] 4/10 used
  Seq 1: [######______] 6/10 used
  Seq 2: [###_________] 3/10 used

After Seq 1 finishes:
  Seq 0: [####________] 4/10 used
  Seq 1: [FREE______FREE] ← Can't be used by Seq 0 or 2!
  Seq 2: [###_________] 3/10 used
```

**Result**: 10 tokens of wasted memory that no one can use.

#### 3. **No Sharing**

Each sequence has its own copy, even when sharing is possible:

```
Beam Search (3 beams from same prefix):
  Beam 0: [K₀, K₁, K₂, K₃, K₄, K₅]  ← All 3 beams share
  Beam 1: [K₀, K₁, K₂, K₃, K₄, K₆]  ← first 5 tokens
  Beam 2: [K₀, K₁, K₂, K₃, K₄, K₇]  ← but must store 3 copies!

Total memory: 3 × 6 = 18 token slots
Could be: 5 (shared) + 3 (unique) = 8 token slots
Waste: 10 token slots (55% overhead!)
```

---

## The Solution

### Virtual Memory Analogy

Paged Attention is inspired by **virtual memory** in operating systems:

| Operating System | Paged Attention |
|-----------------|-----------------|
| Virtual address space | Logical sequence positions |
| Physical memory pages | KV cache blocks |
| Page table | Block table |
| Page fault | Block allocation |
| Copy-on-write | Beam search sharing |

### Core Idea

1. **Divide** KV cache into fixed-size blocks (e.g., 16 tokens)
2. **Map** logical positions → physical blocks using a table
3. **Allocate** blocks on-demand (no pre-allocation)
4. **Share** blocks between sequences when possible

---

## How It Works

### Block-Based Storage

Instead of contiguous memory:

```python
# OLD: Contiguous per sequence
k_cache_seq0 = torch.zeros(n_head, seq_len, head_dim)

# NEW: Blocks that can be anywhere
k_cache_blocks = torch.zeros(num_blocks, n_head, block_size, head_dim)
#                            ↑ Physical  ↑ Per block
```

### Block Table Mapping

Each sequence has a **block table** mapping logical → physical blocks:

```
Logical sequence (15 tokens, block_size=4):
  Positions: [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14]
             └─ blk 0 ┘└─ blk 1 ┘└─ blk 2  ┘└ blk 3┘

Physical memory (8 blocks available):
  Block 0: [empty]
  Block 1: [empty]
  Block 2: [K₀, K₁, K₂, K₃]      ← Logical block 0
  Block 3: [empty]
  Block 4: [K₄, K₅, K₆, K₇]      ← Logical block 1
  Block 5: [K₈, K₉, K₁₀, K₁₁]    ← Logical block 2
  Block 6: [K₁₂, K₁₃, K₁₄, _]    ← Logical block 3 (partial)
  Block 7: [empty]

Block table for this sequence:
  [2, 4, 5, 6]  ← Maps logical → physical
   ↑  ↑  ↑  ↑
   0  1  2  3  (logical block indices)
```

### Accessing a Token

To get K, V for position 9:

```python
position = 9
block_size = 4

# 1. Find logical block
logical_block = position // block_size  # 9 // 4 = 2

# 2. Offset within block
offset = position % block_size  # 9 % 4 = 1

# 3. Look up physical block
physical_block = block_table[logical_block]  # block_table[2] = 5

# 4. Access the cache
k = k_cache[physical_block, :, offset, :]  # Block 5, offset 1
```

### Appending New Tokens

When generating a new token:

```python
# Current state: 10 tokens in blocks [2, 4, 5]
#   Block 2: [K₀, K₁, K₂, K₃]   (full)
#   Block 4: [K₄, K₅, K₆, K₇]   (full)
#   Block 5: [K₈, K₉, _, _]     (2/4 filled)

# Generate token 10 (K₁₀)
position = 10
logical_block = 10 // 4 = 2  # Need logical block 2
physical_block = block_table[2] = 5  # Already allocated!
offset = 10 % 4 = 2

# Write to existing block
k_cache[5, :, 2, :] = K₁₀

# Result:
#   Block 5: [K₈, K₉, K₁₀, _]  (3/4 filled)
```

When we fill a block, allocate a new one:

```python
# Current: 12 tokens in blocks [2, 4, 5]
#   Block 5: [K₈, K₉, K₁₀, K₁₁]  (full!)

# Generate token 12 (K₁₂)
position = 12
logical_block = 12 // 4 = 3  # Need NEW logical block!

# Allocate a free physical block
physical_block = allocator.allocate()  # Returns block 6
block_table.append(6)  # block_table = [2, 4, 5, 6]

# Write to new block
offset = 12 % 4 = 0
k_cache[6, :, 0, :] = K₁₂

# Result:
#   Block 6: [K₁₂, _, _, _]  (1/4 filled)
```

---

## Implementation Details

Our implementation consists of 4 main components:

### 1. BlockTable

Maps logical block indices to physical block IDs.

```python
class BlockTable:
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.blocks = []  # Physical block IDs

    def add_block(self, block_id: int):
        """Allocate a new block."""
        self.blocks.append(block_id)

    def get_block_id(self, logical_idx: int) -> int:
        """Get physical block for logical index."""
        return self.blocks[logical_idx]
```

**Example usage**:
```python
table = BlockTable(block_size=4)
table.add_block(2)  # Logical block 0 → Physical block 2
table.add_block(5)  # Logical block 1 → Physical block 5

assert table.get_block_id(0) == 2
assert table.get_block_id(1) == 5
```

### 2. BlockAllocator

Manages the pool of free blocks (like a memory allocator).

```python
class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.free_blocks = list(range(num_blocks))  # [0,1,2,3,4,5,6,7]
        self.allocated_blocks = set()

    def allocate(self) -> Optional[int]:
        """Allocate a free block."""
        if not self.free_blocks:
            return None  # OOM!
        block_id = self.free_blocks.pop(0)
        self.allocated_blocks.add(block_id)
        return block_id

    def free(self, block_id: int):
        """Return block to free pool."""
        self.allocated_blocks.remove(block_id)
        self.free_blocks.append(block_id)
```

**Example usage**:
```python
allocator = BlockAllocator(num_blocks=8)

# Allocate blocks
b0 = allocator.allocate()  # Returns 0
b1 = allocator.allocate()  # Returns 1

# Free a block
allocator.free(b0)  # Block 0 back to free list

# Can allocate again
b2 = allocator.allocate()  # Returns 0 (reused!)
```

### 3. PagedKVCache

The actual KV cache storage using blocks.

```python
class PagedKVCache:
    def __init__(self, num_blocks, block_size, num_heads, head_dim):
        # Physical storage: all blocks upfront
        self.k_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim)
        self.v_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim)

        # Allocator
        self.allocator = BlockAllocator(num_blocks)

        # Per-sequence metadata
        self.block_tables = {}  # seq_id → BlockTable
        self.sequence_lengths = {}  # seq_id → int

    def allocate_sequence(self, seq_id: int):
        """Create block table for new sequence."""
        self.block_tables[seq_id] = BlockTable(self.block_size)
        self.sequence_lengths[seq_id] = 0

    def append_tokens(self, seq_id: int, k_new, v_new):
        """Add new K, V tokens to sequence."""
        # Calculate which blocks we need
        # Allocate new blocks if necessary
        # Write tokens to appropriate blocks
        # (See full implementation for details)

    def get_kv_cache(self, seq_id: int):
        """Retrieve full K, V by assembling blocks."""
        # Gather blocks in order
        # Concatenate into contiguous tensors
        # Return (k, v)
```

### 4. PagedAttention

Attention layer that uses PagedKVCache.

```python
class PagedAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size=16):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.paged_cache = None  # Initialize later

    def init_paged_cache(self, num_blocks, device):
        self.paged_cache = PagedKVCache(
            num_blocks, self.block_size,
            self.n_head, self.head_dim,
            device=device
        )

    def forward(self, x, seq_id=None, use_paged_cache=False):
        # Compute Q, K, V
        q, k, v = self.compute_qkv(x)

        if use_paged_cache:
            # Append new K, V to cache
            self.paged_cache.append_tokens(seq_id, k, v)

            # Retrieve full cached K, V
            k_full, v_full = self.paged_cache.get_kv_cache(seq_id)

            # Use full K, V for attention
            return self.attention(q, k_full, v_full)
        else:
            return self.attention(q, k, v)
```

---

## Comparison with Standard KV Cache

### Memory Usage

**Scenario**: Serving 4 sequences with varying lengths

| Sequence | Length | Standard Cache (max=100) | Paged Cache (block=16) |
|----------|--------|-------------------------|----------------------|
| Seq 0    | 23     | 100 tokens              | 2 blocks = 32 tokens |
| Seq 1    | 67     | 100 tokens              | 5 blocks = 80 tokens |
| Seq 2    | 45     | 100 tokens              | 3 blocks = 48 tokens |
| Seq 3    | 12     | 100 tokens              | 1 block  = 16 tokens |
| **Total**| **147**| **400 tokens**          | **176 tokens**       |

**Savings**: 400 - 176 = 224 tokens (56% reduction!)

### Detailed Breakdown

```
Standard Cache:
  Each sequence: 100 tokens × 12 layers × 16 heads × 64 head_dim × 2 (K+V) × 2 bytes (fp16)
  = 100 × 12 × 16 × 64 × 2 × 2 = 4.91 MB per sequence
  Total (4 seqs): 19.66 MB
  Actual usage: 147/400 = 36.75%
  Waste: 63.25%

Paged Cache:
  Allocated blocks: 11 blocks × 16 tokens × 12 layers × ...
  = 176 tokens worth of storage
  Actual usage: 147/176 = 83.52%
  Waste: 16.48% (only partially filled last blocks)
```

### Fragmentation Comparison

**Standard cache** after Seq 1 completes:
```
Seq 0: [####################____...] 23/100 used
Seq 1: [FREE__________________FREE] 100 tokens wasted
Seq 2: [###################_____...] 45/100 used
Seq 3: [############________...____] 12/100 used

Total waste: 100 (freed) + 77 + 55 + 88 = 320 tokens (80%)
```

**Paged cache** after Seq 1 completes:
```
Physical blocks:
  Block 0: [Seq0]  Block 1: [Seq0]  Block 2: [FREE]  Block 3: [FREE]
  Block 4: [FREE]  Block 5: [FREE]  Block 6: [FREE]  Block 7: [Seq2]
  Block 8: [Seq2]  Block 9: [Seq2]  Block 10: [Seq3]

After freeing Seq 1's blocks (2,3,4,5,6):
  All blocks return to free pool → can be reused immediately!

Waste: Only partially filled blocks:
  - Seq 0 block 1: 7/16 tokens unused
  - Seq 2 block 9: 11/16 tokens unused
  - Seq 3 block 10: 4/16 tokens unused
  Total: 22 tokens (12.5%)
```

---

## Usage Examples

### Basic Generation

```python
from gpt.paged_attention import PagedAttention

# Initialize
attn = PagedAttention(n_embd=768, n_head=12, block_size=16)
attn.init_paged_cache(num_blocks=1000, device='cuda')

# Allocate sequence
seq_id = 0
attn.paged_cache.allocate_sequence(seq_id)

# Generation loop
for i in range(max_tokens):
    x = embed(current_tokens)  # [1, T, 768]

    # Forward with paged cache
    output = attn(x, seq_id=seq_id, use_paged_cache=True)

    # Sample next token
    next_token = sample(output[:, -1, :])
    current_tokens = torch.cat([current_tokens, next_token], dim=1)

# Clean up
attn.paged_cache.free_sequence(seq_id)
```

### Batch Serving

```python
# Serve multiple sequences concurrently
sequences = {
    0: {"tokens": [1, 2, 3], "finished": False},
    1: {"tokens": [4, 5], "finished": False},
    2: {"tokens": [6, 7, 8, 9], "finished": False},
}

# Allocate all sequences
for seq_id in sequences:
    attn.paged_cache.allocate_sequence(seq_id)

# Generation loop
while any(not s["finished"] for s in sequences.values()):
    for seq_id, seq_data in sequences.items():
        if seq_data["finished"]:
            continue

        # Process this sequence
        x = embed(seq_data["tokens"][-1:])  # Only new token
        output = attn(x, seq_id=seq_id, use_paged_cache=True)

        # Sample and append
        next_token = sample(output[:, -1, :])
        seq_data["tokens"].append(next_token.item())

        # Check if done
        if next_token == EOS:
            seq_data["finished"] = True
            attn.paged_cache.free_sequence(seq_id)  # Free immediately!
```

### Beam Search with Sharing

```python
# Parent sequence
parent_id = 0
attn.paged_cache.allocate_sequence(parent_id)

# Generate prefix
prefix_tokens = generate_prefix(parent_id, prefix_length=20)

# Fork into beams (shares the prefix!)
beam_ids = [1, 2, 3]
for beam_id in beam_ids:
    attn.paged_cache.fork_sequence(parent_id, beam_id)

# Each beam now shares the first 20 tokens (5 blocks if block_size=4)
# Only new divergent tokens allocate new blocks

# Generate for each beam
for beam_id in beam_ids:
    for _ in range(10):  # 10 more tokens
        x = embed(current_tokens[beam_id][-1:])
        output = attn(x, seq_id=beam_id, use_paged_cache=True)
        # ...

# Memory savings:
#   Without sharing: 3 beams × 30 tokens = 90 token slots
#   With sharing: 20 (shared) + 3 × 10 (unique) = 50 token slots
#   Savings: 40 token slots (44%)
```

---

## Performance Analysis

### Memory Savings

From the vLLM paper, serving real workloads:

| Metric | Standard KV Cache | Paged Attention | Improvement |
|--------|------------------|-----------------|-------------|
| Memory utilization | ~40% | ~90% | **2.25x** |
| Sequences served | 100 | 225 | **2.25x** |
| Throughput | 1.0x | 2.2x | **2.2x** |

### Computational Overhead

**Block lookup**: O(1) with block table
**Memory copy**: None (zero-copy gathering)
**Attention**: Same complexity as standard

**Overhead**: < 1% compared to attention computation

### Block Size Trade-offs

| Block Size | Internal Fragmentation | Table Size | Recommendation |
|------------|----------------------|------------|----------------|
| 4          | 12.5% avg           | Large      | Too small      |
| 16         | 3.1% avg            | Medium     | **Good**       |
| 64         | 0.8% avg            | Small      | Too large      |
| 256        | 0.2% avg            | Tiny       | **For long contexts** |

**Rule of thumb**:
- Short sequences (< 100 tokens): block_size = 16
- Medium sequences (100-1000): block_size = 32-64
- Long sequences (> 1000): block_size = 128-256

---

## Advanced Topics

### Copy-on-Write for Beam Search

When forking a sequence:

```python
# Parent has blocks [0, 1, 2]
parent_table = [0, 1, 2]

# Child initially shares
child_table = [0, 1, 2]  # Same blocks!

# When child writes to shared block:
if block_is_shared(block_id):
    new_block = allocator.allocate()  # Get new block
    copy_block(block_id, new_block)   # Copy content
    child_table[idx] = new_block      # Update table
```

### Prefix Caching

Share common prefixes across requests:

```python
# System prompt (same for all requests)
system_blocks = [0, 1, 2]  # Blocks for "You are a helpful assistant..."

# User request 1
user1_table = [0, 1, 2, 3, 4]  # Shares blocks 0,1,2
#                    ↑   ↑  unique blocks

# User request 2
user2_table = [0, 1, 2, 5, 6]  # Also shares 0,1,2
#                    ↑   ↑  different unique blocks

# Memory: 3 (shared) + 2 + 2 = 7 blocks
# vs. standard: 5 + 5 = 10 blocks
```

### Speculative Decoding

Verify multiple tokens simultaneously:

```python
# Draft model proposes: [t1, t2, t3, t4]
# Verifier checks each

# On rejection at t2:
#   - Keep blocks for [t0, t1]
#   - Free blocks for [t2, t3, t4]
#   - Continue from t1

# Easy with paged cache!
table.blocks = table.blocks[:2]  # Keep first 2 blocks
allocator.free(table.blocks[2:])  # Free the rest
```

### Priority-Based Eviction

When out of memory, evict lowest priority sequences:

```python
class PagedKVCache:
    def __init__(self, ...):
        # ...
        self.priorities = {}  # seq_id → priority

    def allocate_with_eviction(self, seq_id, priority):
        while self.allocator.num_free_blocks() == 0:
            # Find lowest priority sequence
            victim = min(self.priorities, key=self.priorities.get)

            if self.priorities[victim] >= priority:
                return False  # Can't evict, OOM

            # Evict victim
            self.free_sequence(victim)

        # Now allocate
        return self.allocate_sequence(seq_id)
```

---

## Summary

### Key Concepts

1. **Block-based storage**: KV cache divided into fixed-size blocks
2. **Block tables**: Map logical positions → physical blocks
3. **On-demand allocation**: Allocate blocks as needed (no pre-allocation)
4. **Sharing**: Multiple sequences can share blocks (copy-on-write)

### Benefits Over Standard KV Cache

| Aspect | Standard | Paged Attention |
|--------|----------|-----------------|
| Memory efficiency | ~40% | ~90% |
| Fragmentation | High | Near-zero |
| Sharing | No | Yes |
| Dynamic growth | No | Yes |
| Throughput | 1x | 2-2.5x |

### When to Use

**Use Paged Attention when:**
- ✅ Serving many concurrent sequences
- ✅ Sequences have varying lengths
- ✅ Memory is limited
- ✅ Need beam search/speculative decoding
- ✅ Production serving (vLLM, TGI)

**Use Standard KV Cache when:**
- ✅ Single sequence generation
- ✅ Research/debugging (simpler)
- ✅ Very short sequences (overhead not worth it)

### Production Usage

vLLM (production LLM serving) uses Paged Attention by default and achieves:
- **2-4x higher throughput** vs. standard serving
- **24x higher throughput** vs. naive HuggingFace implementation
- **Near-zero memory waste** from fragmentation

---

## Further Reading

- [vLLM Paper](https://arxiv.org/abs/2309.06180) - Original Paged Attention paper
- [vLLM GitHub](https://github.com/vllm-project/vllm) - Production implementation
- [Blog Post](https://blog.vllm.ai/2023/06/20/vllm.html) - Intuitive explanation
- [Our Implementation](../gpt/paged_attention.py) - Educational PyTorch code

---

## Try It Out

```bash
# Run the demo
python3 -m gpt.paged_attention

# Shows:
# - Block allocation in action
# - Memory efficiency
# - Step-by-step generation
```

See [paged_attention.py](../gpt/paged_attention.py) for the complete implementation with extensive comments!
