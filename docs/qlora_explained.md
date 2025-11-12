# QLoRA Explained: Quantization + LoRA

This document explains what QLoRA is, how it works, and why it's so effective for memory-efficient fine-tuning.

---

## What is QLoRA?

**QLoRA = Quantized Low-Rank Adaptation**

QLoRA combines two powerful techniques:
1. **4-bit NF4 Quantization**: Compress frozen base model weights to 4 bits
2. **LoRA Adapters**: Add small trainable low-rank matrices

**Key insight**: We can quantize the FROZEN base weights to save memory, while keeping the small LoRA adapters in full precision for training.

---

## Why QLoRA?

### The Memory Problem

When fine-tuning GPT-2 (124M parameters):
- **Full fine-tuning**: 124M × 4 bytes = 496 MB (just for weights!)
- Plus gradients, optimizer states, activations → ~8GB+ VRAM needed

### QLoRA Solution

QLoRA reduces memory by 8x:
- **Base weights (quantized)**: 124M × 0.5 bytes (4-bit) = 62 MB
- **LoRA adapters**: 0.3M × 2 bytes (bfloat16) = 0.6 MB
- **Total**: ~63 MB for the model!

This means you can fine-tune GPT-2 on a GPU with just 4GB VRAM.

---

## How QLoRA Works

### Step 1: Quantize Base Model to 4-bit

Normal neural network weights are 32-bit floats (4 bytes each):
```
Weight: 0.123456789 → 32 bits (4 bytes)
```

QLoRA uses **NF4 (4-bit NormalFloat)** quantization:
```
Weight: 0.123456789 → 4 bits (0.5 bytes)
```

**How?** NF4 uses optimal quantization levels for normally-distributed weights:
1. Divide weights into blocks (e.g., 64 values per block)
2. Find the maximum absolute value in each block (this is the "scale")
3. Normalize weights to [-1, 1] by dividing by scale
4. Map each normalized value to the nearest of 16 quantization levels
5. Store: 4-bit indices + 1 scale per block

**Result**: 8x memory reduction with minimal accuracy loss.

### Step 2: Add LoRA Adapters (Full Precision)

On top of the quantized base, we add small LoRA matrices:

```python
# Original: y = W @ x  (W is 4-bit quantized, frozen)
# QLoRA:    y = W @ x + (B @ A) @ x  (A, B are bfloat16, trainable)

class QuantizedLinearWithLoRA:
    def forward(self, x):
        # Dequantize base weights on-the-fly
        W_dequantized = dequantize(W_quantized, scale)

        # Base output (frozen)
        base_out = W_dequantized @ x

        # LoRA output (trainable)
        lora_out = (B @ A) @ x * (alpha / rank)

        return base_out + lora_out
```

### Step 3: Train Only LoRA Parameters

During training:
- **Frozen**: Quantized base weights (never updated, stay in 4-bit)
- **Trainable**: LoRA matrices A and B (bfloat16)
- **Gradients**: Only flow through LoRA (saves memory!)

---

## NF4 Quantization Deep Dive

### Why NF4?

Neural network weights are typically **normally distributed** (mean ≈ 0, bell curve shape).

NF4 uses quantization levels optimized for normal distributions:
```python
# 16 levels chosen to divide N(0,1) into equal probability regions
NF4_LEVELS = [
    -1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
    0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0
]
```

**Why optimal?** Each quantization "bin" contains approximately the same number of weights → minimal information loss.

### Quantization Process

Example with block_size=64:

```
Original weights (64 values):
[0.245, -0.123, 0.891, ..., -0.045]  # 64 × 4 bytes = 256 bytes

Step 1: Find scale (absmax)
scale = max(abs(weights)) = 0.891

Step 2: Normalize to [-1, 1]
normalized = weights / scale
→ [0.275, -0.138, 1.0, ..., -0.050]

Step 3: Quantize to nearest NF4 level (0-15)
indices = [7, 5, 15, ..., 6]  # Each is 4 bits

Result:
- Quantized: 64 × 0.5 bytes = 32 bytes (indices)
- Scale: 1 × 4 bytes = 4 bytes (float32)
- Total: 36 bytes vs 256 bytes → 7x reduction!
```

### Dequantization Process

To use the weights:

```python
# Lookup quantized values
values = NF4_LEVELS[indices]  # [0.246, -0.185, 1.0, ..., -0.091]

# Denormalize using scale
weights_reconstructed = values * scale
→ [0.219, -0.165, 0.891, ..., -0.081]
```

**Precision loss**: Small error per weight, but overall model quality remains high!

---

## QLoRA vs Regular LoRA vs Full Fine-Tuning

| Aspect | Full Fine-Tuning | LoRA | QLoRA |
|--------|------------------|------|-------|
| **Trainable params** | 124M (100%) | 0.3M (0.24%) | 0.3M (0.24%) |
| **Model memory** | 496 MB (float32) | 496 MB (float32) | 62 MB (4-bit) |
| **Total VRAM** | 8GB+ | 4-8GB | 2-4GB |
| **Training speed** | 1x | 1.2x (faster!) | 0.9x (slightly slower) |
| **Quality** | Best (baseline) | ~99% of full | ~98% of full |
| **Checkpoint size** | 496 MB | 0.6 MB | 0.6 MB |

**When to use each:**
- **Full fine-tuning**: When you have lots of VRAM and data
- **LoRA**: When you want speed and efficiency
- **QLoRA**: When you're VRAM-constrained (< 8GB)

---

## Implementation Details

### Memory Breakdown (GPT-2 124M with QLoRA)

```
Component                    Size
─────────────────────────────────
Quantized base weights       62 MB    (124M × 0.5 bytes)
Quantization scales          1 MB     (124M / 64 × 4 bytes)
LoRA matrices (A + B)        0.6 MB   (0.3M × 2 bytes)
─────────────────────────────────
Total model                  ~64 MB

During training:
+ LoRA gradients             0.6 MB
+ LoRA optimizer states      1.2 MB   (AdamW: 2 states per param)
+ Activations (batch)        ~100 MB  (depends on batch size)
─────────────────────────────────
Peak VRAM                    ~200 MB  (vs 8GB for full fine-tuning!)
```

### Trade-offs

**Advantages:**
- ✅ 8x memory reduction
- ✅ Can fine-tune large models on small GPUs
- ✅ Same quality as regular LoRA
- ✅ Tiny checkpoint files (only LoRA adapters)

**Disadvantages:**
- ❌ 10-20% slower training (dequantization overhead)
- ❌ Slower inference (dequantization per forward pass)
- ❌ More complex implementation
- ❌ CPU inference is VERY slow

---

## Code Walkthrough

### 1. Quantization ([gpt/quantization.py](../gpt/quantization.py))

```python
class NF4Quantizer:
    def quantize(self, weight):
        # Divide into blocks
        blocks = weight.view(n_blocks, block_size)

        # Get scale per block (absmax)
        scale = blocks.abs().max(dim=1)

        # Normalize to [-1, 1]
        normalized = blocks / scale

        # Find nearest NF4 quantile (0-15)
        indices = find_nearest(normalized, NF4_QUANTILES)

        return indices, scale  # 4-bit + scale

    def dequantize(self, indices, scale):
        # Lookup NF4 values
        values = NF4_QUANTILES[indices]

        # Denormalize
        return values * scale  # Back to original range
```

### 2. QLoRA Layer ([gpt/qlora.py](../gpt/qlora.py))

```python
class QuantizedLinearWithLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        # Quantized base (frozen)
        self.quantized_linear = QuantizedLinear(...)

        # LoRA adapter (trainable)
        self.lora = LoRALayer(rank=rank)

    def forward(self, x):
        # Base: dequantize and apply
        base_out = self.quantized_linear(x)  # Dequantizes internally

        # LoRA: apply trainable adapter
        lora_out = self.lora(x)

        # Combine
        return base_out + lora_out
```

### 3. Training ([scripts/training/finetune_qlora.py](../scripts/training/finetune_qlora.py))

```python
# Load base model
model = GPT.from_pretrained("gpt2")

# Apply QLoRA: quantize + add LoRA
apply_qlora_to_model(model, rank=8, alpha=16.0, block_size=64)

# Only optimize LoRA parameters
qlora_params = get_qlora_parameters(model)  # ~0.3M params
optimizer = AdamW(qlora_params)

# Train (gradients only flow through LoRA)
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # Only LoRA gets gradients!
    optimizer.step()

# Save only LoRA adapters (~0.6 MB)
save_qlora_weights(model, "qlora.pt")
```

### 4. Inference ([scripts/inference/generate_qlora.py](../scripts/inference/generate_qlora.py))

```python
# Load base model
model = GPT.from_pretrained("gpt2")

# Apply QLoRA structure
apply_qlora_to_model(model, rank=8, alpha=16.0)

# Load trained LoRA adapters
load_qlora_weights(model, "qlora.pt")

# Generate (with dequantization overhead)
output = model.generate(prompt)
```

---

## Practical Tips

### Choosing Block Size

The `block_size` parameter controls quantization granularity:

```python
# Smaller block size (e.g., 32)
- More accurate (more fine-grained scales)
- More memory for scales
- Recommended for: Small models, when accuracy is critical

# Larger block size (e.g., 128)
- Less accurate (coarser scales)
- Less memory for scales
- Recommended for: Large models, when memory is tight

# Default (64): Good balance for most cases
```

### Choosing LoRA Rank

Same principles as regular LoRA:

```python
# rank=4: Very parameter-efficient, may underfit
# rank=8: Good default (used in paper)
# rank=16: More expressive, still efficient
# rank=32+: Approaching full fine-tuning cost
```

### GPU vs CPU

**GPU (CUDA):**
- ✅ Dequantization is fast (parallel)
- ✅ bfloat16 support (if Ampere+)
- Recommended!

**CPU:**
- ❌ Dequantization is slow (serial)
- ❌ No bfloat16 (uses float32)
- Only use for small experiments

---

## Comparison to Other Quantization Methods

### Int8 Quantization

```python
# Int8: Uniform quantization
- 8 bits = 256 levels
- Linear spacing: -127, -126, ..., 0, ..., 126, 127
- Good for: Activations (uniform distribution)
- Memory: 4x reduction

# NF4: Information-theoretic quantization
- 4 bits = 16 levels
- Non-linear spacing: optimized for normal distribution
- Good for: Weights (normal distribution)
- Memory: 8x reduction
```

### Mixed Precision (bfloat16)

```python
# bfloat16: Standard mixed precision
- 16 bits (vs 32-bit float32)
- No quantization (just lower precision)
- Memory: 2x reduction
- Widely supported

# NF4: Extreme quantization
- 4 bits
- Requires custom dequantization
- Memory: 8x reduction
- Requires implementation (like ours!)
```

---

## References and Further Reading

**Original QLoRA Paper:**
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- https://arxiv.org/abs/2305.14314

**Key innovations:**
1. NF4 quantization for normal distributions
2. Double quantization (quantizing the scales too)
3. Paged optimizers (for very large models)

**Our implementation:**
- Simplified for educational purposes
- Implements NF4 quantization from scratch
- No external dependencies (bitsandbytes)
- Clear code structure for learning

**Related concepts:**
- LoRA: [docs/lora_explained.md](lora_explained.md) (if exists)
- Quantization theory: [gpt/quantization.py](../gpt/quantization.py) (detailed comments)
- Training architecture: [docs/training_architecture.md](training_architecture.md)

---

## Summary

**QLoRA in one sentence:**
> QLoRA compresses frozen base weights to 4-bit and adds small trainable LoRA adapters, achieving 8x memory reduction while maintaining quality.

**When to use QLoRA:**
- ✅ Limited VRAM (< 8GB)
- ✅ Want to fine-tune larger models on consumer GPUs
- ✅ Don't mind 10-20% slower training
- ✅ Have a GPU (CPU is too slow)

**Key takeaways:**
1. NF4 quantization is information-theoretically optimal for normal distributions
2. Frozen weights can be quantized aggressively without hurting quality
3. Trainable adapters need full precision
4. Memory vs speed trade-off is worth it for VRAM-constrained scenarios

Enjoy your memory-efficient fine-tuning! 🚀
