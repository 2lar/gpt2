# QLoRA Code Walkthrough: Step-by-Step Application

This document shows **exactly** how QLoRA is applied in your codebase, tracing through the code step-by-step.

---

## The Big Picture: What Happens When You Train

When you run:
```bash
python -m scripts.training.finetune_qlora --data brainrot_data --steps 500
```

Here's the **exact sequence** of what happens:

```
1. Load base GPT-2 model (float32, 124M params)
   ↓
2. Apply QLoRA to model (replaces Linear layers)
   ↓
3. For each Linear layer:
   a. Quantize weights to 4-bit
   b. Add LoRA adapters
   c. Replace original layer with QuantizedLinearWithLoRA
   ↓
4. Train only LoRA parameters
   ↓
5. Save only LoRA weights (not quantized base)
```

Let's trace through each step with actual code!

---

## Step 1: Load Base Model

**File:** [scripts/training/finetune_qlora.py:133](../scripts/training/finetune_qlora.py#L133)

```python
model = GPT.from_pretrained("gpt2", weights_path="data/gpt2_weights.pt")
```

**What this does:**
- Loads GPT-2 architecture (12 layers)
- Loads pretrained weights (float32)
- Each layer has `c_attn` (attention) and `c_proj` (projection) Linear layers

**Before QLoRA, a GPT-2 block looks like:**
```python
Block(
  attn: Attention(
    c_attn: Linear(768 → 2304)      # Q, K, V projections (float32)
    c_proj: Linear(768 → 768)       # Output projection (float32)
  )
  mlp: MLP(
    c_fc: Linear(768 → 3072)        # Feed-forward (float32)
    c_proj: Linear(3072 → 768)      # Feed-forward (float32)
  )
)
```

---

## Step 2: Apply QLoRA

**File:** [scripts/training/finetune_qlora.py:147-152](../scripts/training/finetune_qlora.py#L147-152)

```python
lora_params_count = apply_qlora_to_model(
    model,
    rank=8,
    alpha=16.0,
    target_modules=['c_attn'],  # ← Only replace attention layers!
    block_size=64,
    dtype=torch.bfloat16,
)
```

**What `apply_qlora_to_model()` does:**

### Step 2a: Find Target Layers

**File:** [gpt/qlora.py:165-176](../gpt/qlora.py#L165-176)

```python
def _replace_linear(parent_module, name, target_names):
    for child_name, child_module in parent_module.named_children():
        if isinstance(child_module, nn.Linear):
            # Check if this is 'c_attn'
            if 'c_attn' in child_name:
                # REPLACE THIS LAYER!
                qlora_layer = QuantizedLinearWithLoRA.from_linear(
                    child_module, rank=8, alpha=16.0, block_size=64
                )
                setattr(parent_module, child_name, qlora_layer)
```

**For GPT-2, this finds:**
- Layer 0: `blocks[0].attn.c_attn` ✓ (replace)
- Layer 0: `blocks[0].attn.c_proj` ✗ (skip)
- Layer 0: `blocks[0].mlp.c_fc` ✗ (skip)
- ... (12 layers total)
- **Result:** 12 `c_attn` layers get replaced

---

## Step 3: Convert Each Linear to QLoRA

**File:** [gpt/qlora.py:100-137](../gpt/qlora.py#L100-137)

For each `c_attn` layer, here's what `from_linear()` does:

### Step 3a: Create QuantizedLinearWithLoRA Wrapper

```python
qlora_layer = QuantizedLinearWithLoRA(
    in_features=768,      # From original layer
    out_features=2304,    # From original layer
    rank=8,
    alpha=16.0,
    bias=True,
    block_size=64,
    dtype=torch.bfloat16,
)
```

**This creates TWO sub-components:**

#### Component 1: QuantizedLinear (Frozen Base)

**File:** [gpt/quantization.py:154-184](../gpt/quantization.py#L154-184)

```python
self.quantized_linear = QuantizedLinear(
    in_features=768,
    out_features=2304,
    bias=True,
    block_size=64,
)

# Stores:
# - weight_quantized: uint8 tensor (4-bit indices, 0-15)
# - weight_scale: float32 tensor (one per block)
# - bias: bfloat16 tensor (full precision)
```

#### Component 2: LoRA Layer (Trainable Adapters)

**File:** [gpt/lora.py](../gpt/lora.py) (imported)

```python
self.lora = LoRALayer(
    in_features=768,
    out_features=2304,
    rank=8,
    alpha=16.0,
)

# Creates:
# - lora_A: [rank, in_features] = [8, 768] = 6,144 params
# - lora_B: [out_features, rank] = [2304, 8] = 18,432 params
# Total: 24,576 trainable params per c_attn layer
```

### Step 3b: Quantize Original Weights

**File:** [gpt/qlora.py:131](../gpt/qlora.py#L131)

```python
# Take original float32 weights [2304, 768]
original_weights = linear.weight.data  # Shape: [2304, 768]

# Quantize to 4-bit
qlora_layer.quantized_linear.quantize_weights(original_weights)
```

**What `quantize_weights()` does:**

**File:** [gpt/quantization.py:44-73](../gpt/quantization.py#L44-73)

```python
def quantize_weights(self, weight: torch.Tensor):
    """
    Input: [2304, 768] float32 tensor
    """
    # Step 1: Flatten
    weight_flat = weight.flatten()  # [1,769,472] values

    # Step 2: Divide into blocks of 64
    n_blocks = 1,769,472 / 64 = 27,648 blocks
    weight_blocks = weight_flat.view(27648, 64)

    # Step 3: Get scale per block (absmax)
    absmax = weight_blocks.abs().max(dim=1)  # [27,648] scales

    # Step 4: Normalize to [-1, 1]
    normalized = weight_blocks / absmax  # [27648, 64]

    # Step 5: Find nearest NF4 quantile (0-15)
    # NF4_QUANTILES = [-1.0, -0.696, ..., 0.0, ..., 1.0]  (16 levels)
    distances = |normalized - each quantile|
    quantized = argmin(distances)  # [27648, 64] with values 0-15

    # Store results
    self.weight_quantized = quantized.flatten()  # uint8 (4-bit values)
    self.weight_scale = absmax                   # float32 scales
    self.weight_shape = torch.Size([2304, 768]) # Original shape
```

**Memory saved:**
- Original: 2304 × 768 × 4 bytes (float32) = 7,077,888 bytes ≈ **6.75 MB**
- Quantized:
  - Indices: 2304 × 768 × 0.5 bytes (4-bit) = 884,736 bytes
  - Scales: 27,648 × 4 bytes (float32) = 110,592 bytes
  - Total: 995,328 bytes ≈ **0.95 MB**
- **Savings: ~7x reduction per layer!**

---

## Step 4: After QLoRA is Applied

**The model structure now looks like:**

```python
Block(
  attn: Attention(
    c_attn: QuantizedLinearWithLoRA(     # ← REPLACED!
      quantized_linear: QuantizedLinear(
        weight_quantized: [1769472] uint8  # 4-bit indices
        weight_scale: [27648] float32      # Scales
        bias: [2304] bfloat16             # Bias
      )
      lora: LoRALayer(
        lora_A: [8, 768] bfloat16         # Trainable!
        lora_B: [2304, 8] bfloat16        # Trainable!
      )
    )
    c_proj: Linear(768 → 768)            # ← NOT replaced (still float32)
  )
  mlp: MLP(...)                          # ← NOT replaced
)
```

**Total for all 12 layers:**
- Quantized `c_attn` weights: 12 × 0.95 MB ≈ 11.4 MB
- LoRA adapters: 12 × 24,576 × 2 bytes ≈ 0.6 MB
- Other layers (c_proj, MLP): Still float32 ≈ 50 MB
- **Total model: ~62 MB vs 496 MB original**

---

## Step 5: Forward Pass During Training

When you run `model(x)` during training, here's what happens:

**File:** [gpt/qlora.py:80-97](../gpt/qlora.py#L80-97)

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x shape: [batch, seq_len, 768]

    # 1. Dequantize and apply base layer (FROZEN)
    base_output = self.quantized_linear(x)

    # 2. Apply LoRA adapter (TRAINABLE)
    lora_output = self.lora(x)

    # 3. Combine
    return base_output + lora_output
```

### Step 5a: Dequantize Base Weights

**File:** [gpt/quantization.py:186-203](../gpt/quantization.py#L186-203)

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Dequantize on-the-fly (this happens EVERY forward pass!)
    weight_fp = self.dequantizer.dequantize(
        self.weight_quantized,  # [1769472] uint8 (4-bit indices)
        self.weight_scale,       # [27648] float32 (scales)
        self.weight_shape,       # [2304, 768]
        dtype=torch.bfloat16
    )

    # Now weight_fp is [2304, 768] bfloat16
    # Apply linear transformation
    return F.linear(x, weight_fp, self.bias)
```

**Dequantization process:**

**File:** [gpt/quantization.py:74-101](../gpt/quantization.py#L74-101)

```python
def dequantize(self, quantized, scale, original_shape):
    # quantized: [1769472] with values 0-15

    # Step 1: Reshape into blocks
    quantized_blocks = quantized.view(27648, 64)

    # Step 2: Map indices to NF4 values
    # NF4_QUANTILES[0] = -1.0, NF4_QUANTILES[15] = 1.0, etc.
    dequantized = NF4_QUANTILES[quantized_blocks]  # [27648, 64]

    # Step 3: Denormalize using scales
    dequantized = dequantized * scale.unsqueeze(-1)  # [27648, 64]

    # Step 4: Flatten and reshape
    dequantized = dequantized.flatten()[:1769472]
    return dequantized.view(2304, 768)  # bfloat16
```

### Step 5b: Apply LoRA Adapter

**File:** [gpt/lora.py](../gpt/lora.py)

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [batch, seq_len, 768]
    # lora_A: [8, 768]
    # lora_B: [2304, 8]

    # Apply low-rank transformation
    result = (x @ lora_A.T) @ lora_B.T  # [batch, seq_len, 2304]

    # Scale by alpha/rank
    result = result * (self.alpha / self.rank)  # (16.0 / 8) = 2.0

    return result
```

### Step 5c: Combine Outputs

```python
# Base: Dequantized frozen weights
base_output = quantized_linear(x)  # [batch, seq_len, 2304]

# LoRA: Trainable adapters
lora_output = lora(x)              # [batch, seq_len, 2304]

# Final output
output = base_output + lora_output  # Element-wise addition!
```

**Key insight:** The base weights are FROZEN (no gradients), but LoRA weights get gradients!

---

## Step 6: Backward Pass (Gradients)

During `loss.backward()`:

```python
# Gradient flow:
loss
  ↓
output = base_output + lora_output
  ↓                    ↓
  ✗                    ✓ (gradients flow here!)
  |                    |
  |                  lora_A, lora_B (UPDATED!)
  |
quantized_linear (FROZEN, no gradients)
```

**File:** [scripts/training/finetune_qlora.py:235](../scripts/training/finetune_qlora.py#L235)

```python
loss.backward()  # Only computes gradients for lora_A and lora_B!
```

**Why base weights don't get gradients:**
- They're stored in buffers (`register_buffer`), not parameters
- PyTorch doesn't compute gradients for buffers
- Saves memory (no gradient tensors for 124M params!)

---

## Step 7: Optimizer Update

**File:** [scripts/training/finetune_qlora.py:193-202](../scripts/training/finetune_qlora.py#L193-202)

```python
# Get only LoRA parameters
qlora_params = get_qlora_parameters(model)  # Only lora_A and lora_B!

# Optimizer only updates these
optimizer = torch.optim.AdamW(qlora_params, lr=3e-4)

# During training
optimizer.step()  # Updates only lora_A and lora_B
```

**What `get_qlora_parameters()` returns:**

**File:** [gpt/qlora.py:201-213](../gpt/qlora.py#L201-213)

```python
def get_qlora_parameters(model: nn.Module) -> List[nn.Parameter]:
    lora_params = []

    for module in model.modules():
        if isinstance(module, QuantizedLinearWithLoRA):
            # Only LoRA matrices are trainable!
            lora_params.append(module.lora.lora_A)
            lora_params.append(module.lora.lora_B)

    return lora_params
```

**Total trainable parameters:**
- 12 layers × 2 matrices × 24,576 params = **589,824 params**
- That's only **0.48%** of the full 124M model!

---

## Step 8: Save Checkpoint

**File:** [scripts/training/finetune_qlora.py:275-276](../scripts/training/finetune_qlora.py#L275-276)

```python
save_qlora_weights(model, "qlora_checkpoints/qlora_final.pt")
```

**What gets saved:**

**File:** [gpt/qlora.py:215-232](../gpt/qlora.py#L215-232)

```python
def save_qlora_weights(model: nn.Module, path: str):
    lora_state = {}

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinearWithLoRA):
            # Save ONLY LoRA matrices (not quantized base!)
            lora_state[f"{name}.lora_A"] = module.lora.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora.lora_B.data.cpu()

    torch.save(lora_state, path)
```

**Checkpoint contents:**
```python
{
  'blocks.0.attn.c_attn.lora_A': tensor([8, 768]),
  'blocks.0.attn.c_attn.lora_B': tensor([2304, 8]),
  'blocks.1.attn.c_attn.lora_A': tensor([8, 768]),
  'blocks.1.attn.c_attn.lora_B': tensor([2304, 8]),
  ...  # 12 layers total
}
```

**Checkpoint size:** 589,824 params × 2 bytes (bfloat16) ≈ **1.1 MB**

---

## Step 9: Load for Inference

**File:** [scripts/inference/generate_qlora.py:165-176](../scripts/inference/generate_qlora.py#L165-176)

```python
# 1. Load base model (float32)
model = GPT.from_pretrained("gpt2", weights_path="data/gpt2_weights.pt")

# 2. Apply QLoRA structure (quantize + add LoRA slots)
apply_qlora_to_model(model, rank=8, alpha=16.0, block_size=64)

# 3. Load trained LoRA weights
load_qlora_weights(model, "qlora_checkpoints/qlora_final.pt")

# 4. Generate
output = model.generate(prompt)
```

**During generation:**
- Each forward pass dequantizes base weights (overhead!)
- LoRA adapters are already in memory (bfloat16)
- Outputs are identical quality to regular LoRA

---

## Summary: The Complete Flow

```
Training:
1. Load base model (float32, 496 MB)
2. Apply QLoRA:
   - Quantize base weights → 4-bit (62 MB)
   - Add LoRA adapters → bfloat16 (0.6 MB)
3. Train only LoRA (gradients only flow through adapters)
4. Save only LoRA weights (1.1 MB checkpoint)

Inference:
1. Load base model (float32, 496 MB)
2. Apply QLoRA structure (quantize → 62 MB)
3. Load LoRA weights (1.1 MB)
4. Generate:
   - Dequantize base weights per forward pass
   - Add LoRA adaptation
   - Output text
```

---

## Key Differences from Regular LoRA

| Aspect | Regular LoRA | QLoRA |
|--------|--------------|-------|
| **Base weights** | float32 (4 bytes) | 4-bit NF4 (0.5 bytes) |
| **Forward pass** | Direct matrix multiply | Dequantize → multiply |
| **Memory** | 496 MB base + 0.6 MB LoRA | 62 MB base + 0.6 MB LoRA |
| **Speed** | Faster (no dequant) | 10-20% slower |
| **Quality** | Baseline | ~99% of LoRA |

---

## Questions Answered

**Q: Where does quantization happen?**
A: In `QuantizedLinearWithLoRA.from_linear()` → calls `quantize_weights()`

**Q: Where does dequantization happen?**
A: Every forward pass in `QuantizedLinear.forward()` → calls `dequantize()`

**Q: Why is it slower?**
A: Dequantization happens on-the-fly (CPU→GPU transfer + lookup + denormalize)

**Q: What gets trained?**
A: Only `lora_A` and `lora_B` matrices (0.6M params). Base weights are frozen buffers.

**Q: What gets saved in checkpoint?**
A: Only LoRA matrices. You need original base weights to load the model.

**Q: How does it save memory?**
A: 8x reduction on base weights (4-bit vs 32-bit), plus no gradients for base weights.

---

Now you understand **exactly** how QLoRA works in your codebase! 🎉
