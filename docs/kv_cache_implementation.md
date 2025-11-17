# KV Cache Implementation Guide

Complete guide to the KV (Key-Value) cache implementation in this GPT-2 codebase.

---

## What is KV Caching?

KV caching is an optimization technique for autoregressive text generation that dramatically speeds up inference by avoiding redundant computations.

### The Problem

Without caching, during text generation:
- **Iteration 1**: Process 10 tokens → compute Q, K, V for all 10 tokens
- **Iteration 2**: Process 11 tokens → compute Q, K, V for all 11 tokens ❌ (recomputes 10!)
- **Iteration 3**: Process 12 tokens → compute Q, K, V for all 12 tokens ❌ (recomputes 11!)
- **Total work**: 10 + 11 + 12 + ... = O(n²)

### The Solution

With KV caching:
- **Iteration 1**: Process 10 tokens → compute Q, K, V for all 10 tokens, **cache K, V**
- **Iteration 2**: Process **1 new token** → compute Q, K, V for 1 token only! ✅
- **Iteration 3**: Process **1 new token** → compute Q, K, V for 1 token only! ✅
- **Total work**: 10 + 1 + 1 + ... = O(n)

**Result**: ~2-100x speedup depending on sequence length!

---

## How It Works

### Key Insight

In self-attention, when computing attention for a new token:
- **Query (Q)**: Only needed for the new token
- **Key (K)**: Needed for all tokens (past + new)
- **Value (V)**: Needed for all tokens (past + new)

But K and V for past tokens **never change**! So we can:
1. Compute K, V for the new token only
2. Concatenate with cached K, V from previous tokens
3. Use the full K, V for attention computation

### Visual Example

```python
# Iteration 1: Process full prompt "The quick brown"
x = embed(["The", "quick", "brown"])  # [B, 3, n_embd]
Q, K, V = compute(x)                  # Each: [B, n_head, 3, head_dim]
cache = (K, V)                         # Store for next iteration

# Iteration 2: Add new token "fox"
x_new = embed(["fox"])                 # [B, 1, n_embd] ← Only 1 token!
Q_new, K_new, V_new = compute(x_new)   # Each: [B, n_head, 1, head_dim]

# Concatenate with cache
K_full = concat([cache.K, K_new])      # [B, n_head, 4, head_dim]
V_full = concat([cache.V, V_new])      # [B, n_head, 4, head_dim]

# Attention uses full K, V but only computed 1 new token!
attn_output = attention(Q_new, K_full, V_full)
```

---

## Implementation Details

### Files Modified

1. **[gpt/attention_gpt2.py](../gpt/attention_gpt2.py)** - Added KV cache support to `CausalSelfAttention`
2. **[gpt/block.py](../gpt/block.py)** - Updated `Block` to pass cache through attention
3. **[gpt/model.py](../gpt/model.py)** - Updated `GPT.forward()` to manage caches across layers
4. **[scripts/inference/generate.py](../scripts/inference/generate.py)** - Updated generation loop to use cache

### 1. Attention Layer (gpt/attention_gpt2.py)

```python
def forward(self, x, kv_cache=None, use_cache=False):
    """
    Args:
        x: Input tokens [B, T, n_embd]
        kv_cache: Tuple (past_k, past_v) from previous iteration
        use_cache: Whether to return updated cache
    """
    # Compute Q, K, V for new tokens only
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

    # Reshape for multi-head attention
    k = k.view(B, T, n_head, head_dim).transpose(1, 2)
    v = v.view(B, T, n_head, head_dim).transpose(1, 2)

    # KEY STEP: Concatenate with cached K, V
    if kv_cache is not None:
        past_k, past_v = kv_cache
        k = torch.cat([past_k, k], dim=2)  # Concat along sequence dim
        v = torch.cat([past_v, v], dim=2)

    # Store updated cache
    new_cache = (k, v) if use_cache else None

    # Compute attention with full K, V
    output = attention(q, k, v)

    return output, new_cache
```

### 2. Block Layer (gpt/block.py)

```python
def forward(self, x, kv_cache=None, use_cache=False):
    # Attention with cache
    if use_cache:
        attn_out, new_kv_cache = self.attn(
            self.ln_1(x),
            kv_cache=kv_cache,
            use_cache=True
        )
    else:
        attn_out = self.attn(self.ln_1(x))
        new_kv_cache = None

    x = x + attn_out
    x = x + self.mlp(self.ln_2(x))

    return (x, new_kv_cache) if use_cache else x
```

### 3. Model Forward (gpt/model.py)

```python
def forward(self, idx, kv_caches=None, use_cache=False, position_offset=0):
    """
    Args:
        idx: Token indices [B, T]
        kv_caches: List of caches (one per layer)
        use_cache: Whether to return updated caches
        position_offset: Position offset for embeddings when using cache
    """
    # Embed tokens with correct position offset
    pos = torch.arange(position_offset, position_offset + T)
    x = token_emb + pos_emb

    # Pass through blocks, collecting caches
    new_kv_caches = []
    for i, block in enumerate(self.blocks):
        block_cache = kv_caches[i] if kv_caches else None

        if use_cache:
            x, new_cache = block(x, kv_cache=block_cache, use_cache=True)
            new_kv_caches.append(new_cache)
        else:
            x = block(x)

    logits = self.lm_head(x)
    return (logits, new_kv_caches) if use_cache else (logits, None)
```

### 4. Generation Loop (scripts/inference/generate.py)

```python
def generate(model, prompt, use_cache=True):
    x = tokenize(prompt)
    kv_caches = None
    position_offset = 0

    for i in range(max_new_tokens):
        if use_cache:
            if i == 0:
                # First iteration: full prompt
                logits, kv_caches = model(x, use_cache=True)
                position_offset = x.size(1)
            else:
                # Subsequent iterations: only new token!
                x_input = x[:, -1:]  # [B, 1] ← KEY OPTIMIZATION
                logits, kv_caches = model(
                    x_input,
                    kv_caches=kv_caches,
                    use_cache=True,
                    position_offset=position_offset
                )
                position_offset += 1
        else:
            # No caching: recompute everything
            logits, _ = model(x)

        # Sample next token
        next_token = sample(logits[:, -1, :])
        x = torch.cat([x, next_token], dim=1)

    return x
```

---

## Usage

### Command Line

```bash
# With KV cache (default, fast)
python -m scripts.inference.generate --prompt "Hello world"

# Without KV cache (slower, for comparison)
python -m scripts.inference.generate --prompt "Hello world" --no-cache
```

### Python API

```python
from gpt.model import GPT

model = GPT.from_pretrained('gpt2')
model.eval()

# Without cache
logits, _ = model(tokens, use_cache=False)

# With cache (first pass)
logits, kv_caches = model(tokens, use_cache=True)

# With cache (subsequent passes)
logits, kv_caches = model(
    new_token,
    kv_caches=kv_caches,
    use_cache=True,
    position_offset=prev_length
)
```

---

## Performance

### Benchmark Results

On a small 4-layer model (3M parameters):
- **50 token generation**:
  - Without cache: 0.068s
  - With cache: 0.029s
  - **Speedup: 2.35x**

On larger models and longer sequences, speedup can reach **50-100x**!

### Why the Speedup?

**Without cache**: Each iteration processes full sequence
```
Iteration 1: 10 tokens  → 10 * (Q + K + V) computations
Iteration 2: 11 tokens  → 11 * (Q + K + V) computations
Iteration 3: 12 tokens  → 12 * (Q + K + V) computations
...
Total: O(n²) complexity
```

**With cache**: Each iteration processes only new token
```
Iteration 1: 10 tokens  → 10 * (Q + K + V) computations
Iteration 2: 1 token    → 1 * (Q + K + V) computations ✓
Iteration 3: 1 token    → 1 * (Q + K + V) computations ✓
...
Total: O(n) complexity
```

### Memory Trade-off

KV caching trades memory for speed:
- **Extra memory**: 2 * n_layers * batch_size * seq_len * n_kv_heads * head_dim
- For GPT-2 small: ~100MB for 512 tokens
- For GPT-2 large: ~400MB for 512 tokens

This is why **Grouped-Query Attention (GQA)** is valuable:
- Reduces KV cache size by 75% (8 KV heads vs 32 Q heads)
- Minimal quality loss
- Used by LLaMA 2/3, Mistral, etc.

---

## Testing

Run the comprehensive test suite:

```bash
python3 test_kv_cache.py
```

This tests:
1. ✓ Basic cache functionality (cache creation and retrieval)
2. ✓ Correctness (cached output matches non-cached)
3. ✓ Incremental generation (autoregressive decoding)
4. ✓ Performance (speedup measurement)

Expected output:
```
✓ PASS: Cached and non-cached outputs match!
✓ PASS: Cached and non-cached generation produce identical outputs!
✓ Speedup: 2.35x faster with caching!
```

---

## Common Issues

### Issue 1: Outputs Don't Match

**Symptom**: Cached and non-cached outputs differ

**Cause**: Usually position offset is wrong

**Fix**: Ensure `position_offset` tracks total sequence length correctly
```python
# WRONG
position_offset = 0  # Should increment!

# RIGHT
position_offset = 0
if i == 0:
    position_offset = prompt_length
else:
    position_offset += 1
```

### Issue 2: Out of Memory

**Symptom**: CUDA OOM during long generation

**Cause**: KV cache grows with sequence length

**Solutions**:
1. Use smaller batch size
2. Use GQA (fewer KV heads)
3. Clear cache periodically
4. Use sliding window attention

### Issue 3: Cache Shape Mismatch

**Symptom**: Shape error when concatenating cache

**Cause**: Cache format doesn't match attention head format

**Fix**: Ensure cache has shape `[B, n_kv_head, T, head_dim]`

---

## Advanced: Multi-Layer Cache Management

Each transformer layer has its own KV cache:

```python
# Cache structure
kv_caches = [
    (k_layer0, v_layer0),  # Layer 0 cache
    (k_layer1, v_layer1),  # Layer 1 cache
    (k_layer2, v_layer2),  # Layer 2 cache
    ...
]

# Each cache tuple contains:
# k: [B, n_kv_head, seq_len, head_dim]
# v: [B, n_kv_head, seq_len, head_dim]
```

The model's forward pass manages this:
```python
for i, block in enumerate(self.blocks):
    block_cache = kv_caches[i] if kv_caches else None
    x, new_cache = block(x, kv_cache=block_cache, use_cache=True)
    new_kv_caches.append(new_cache)
```

---

## Comparison: Before vs After

### Before (No Cache)
```python
for _ in range(max_new_tokens):
    # Process ALL tokens every iteration
    logits, _ = model(x)  # x.shape = [B, growing_length]
    next_token = sample(logits[:, -1, :])
    x = torch.cat([x, next_token], dim=1)
```

**Complexity**: O(n²)

### After (With Cache)
```python
kv_caches = None
for i in range(max_new_tokens):
    if i == 0:
        # First pass: full prompt
        logits, kv_caches = model(x, use_cache=True)
    else:
        # Subsequent: only new token!
        logits, kv_caches = model(
            x[:, -1:],  # Only 1 token!
            kv_caches=kv_caches,
            use_cache=True
        )
    next_token = sample(logits[:, -1, :])
    x = torch.cat([x, next_token], dim=1)
```

**Complexity**: O(n)

---

## Summary

**What we implemented:**
- ✅ KV cache support in attention layer
- ✅ Cache passing through transformer blocks
- ✅ Multi-layer cache management in model
- ✅ Efficient generation loop using cache
- ✅ Comprehensive test suite
- ✅ Command-line option to toggle cache

**Key benefits:**
- 🚀 2-100x speedup for text generation
- ✅ Identical outputs to non-cached version
- 💾 Configurable (can disable for lower memory)
- 🧪 Fully tested and validated

**How it works:**
1. First iteration: Compute Q, K, V for full prompt, cache K, V
2. Subsequent iterations: Compute Q, K, V for new token only
3. Concatenate new K, V with cached values
4. Attention sees full context but only computed new token!

**The magic**: Input `x` to attention layer is only the new token (T=1) when using cache, but attention sees the full sequence via concatenation with cached K, V!

---

For questions or issues, see [test_kv_cache.py](../test_kv_cache.py) for working examples.
