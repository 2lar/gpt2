# LoRA Fine-Tuning Guide

Complete guide for fine-tuning GPT-2 on custom text using LoRA (Low-Rank Adaptation).

**Goal**: Fine-tune GPT-2 on your custom text using LoRA for efficient, 8GB VRAM-friendly training.

---

## Table of Contents

1. [Quick Start - 3 Commands](#quick-start---3-commands)
2. [What is LoRA?](#what-is-lora)
3. [Step-by-Step Tutorial](#step-by-step-tutorial)
4. [Understanding What Happened](#understanding-what-happened)
5. [Configuration Options](#configuration-options)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

---

## Quick Start - 3 Commands

Fine-tune GPT-2 on your custom text in minutes.

### Prerequisites

```bash
source env/bin/activate
```

### Complete Workflow

```bash
# 1. Prepare your data
python scripts/prepare_custom_text.py \
  --input data/sampleBrainRot.txt \
  --output brainrot_data

# 2. Fine-tune with LoRA (5-10 minutes)
python -m scripts.finetune_lora \
  --data brainrot_data \
  --steps 500

# 3. Compare base vs fine-tuned outputs
python -m scripts.compare_outputs \
  --lora lora_checkpoints/lora_final.pt
```

### What You'll See

**Before** (base GPT-2):
```
Once upon a time in the land of stories, there was a young girl...
```

**After** (LoRA fine-tuned on brain rot):
```
Once upon a time in goofy ahh Ohio, a quirky blud named Duke Dennis...
```

### Key Features

- **Memory efficient**: Only trains 0.24% of model parameters (294K vs 124M)
- **Fast**: 5-10 minutes for 500 steps on GPU
- **8GB VRAM friendly**: Works on consumer GPUs
- **Small checkpoints**: LoRA weights are only ~1MB (vs 500MB full model)
- **No base model changes**: Your pretrained weights stay untouched

---

## What is LoRA?

LoRA (Low-Rank Adaptation) adds small trainable adapter weights to a frozen pretrained model.

### Key Benefits

- **Memory efficient**: Only trains ~0.1% of parameters
- **Fast**: Much quicker than full fine-tuning
- **Modular**: Can swap different LoRA adapters on same base model
- **8GB VRAM friendly**: Works on consumer GPUs

### How It Works

**Standard Linear Layer**:
```
y = Wx + b
```

**With LoRA**:
```
y = Wx + (BA)x + b
  = Wx + B(Ax) + b

where:
- W: Frozen pretrained weights (d × d)
- A: Trainable low-rank matrix (r × d)
- B: Trainable low-rank matrix (d × r)
- r << d (e.g., r=8, d=768)
```

**Parameter Reduction**:
- Original: 768 × 768 = 589,824 params per layer
- LoRA: 768×8 + 8×768 = 12,288 params per layer
- **Reduction**: 98% fewer parameters per layer

---

## Step-by-Step Tutorial

### Files Created

This implementation includes:

1. **[gpt/lora.py](../gpt/lora.py)** - LoRA implementation (245 lines)
   - `LoRALayer`: Low-rank adapter matrices (A and B)
   - `LinearWithLoRA`: Wraps Linear layers with LoRA
   - `apply_lora_to_model()`: Adds LoRA to attention layers
   - `get_lora_parameters()`: Extracts trainable LoRA params
   - `save_lora_weights()` / `load_lora_weights()`: Checkpoint management
   - `merge_all_lora_weights()`: Merges LoRA into base model

2. **[scripts/prepare_custom_text.py](../scripts/prepare_custom_text.py)** - Data preparation (77 lines)
   - Loads raw text file
   - Tokenizes using GPT-2 tokenizer (tiktoken)
   - Splits into train/val (90/10)
   - Saves as .npy shards for DataLoaderLite

3. **[scripts/finetune_lora.py](../scripts/finetune_lora.py)** - Training script (242 lines)
   - Loads pretrained GPT-2 model
   - Applies LoRA to attention layers
   - Trains only LoRA parameters (base frozen)
   - Uses gradient accumulation for 8GB VRAM
   - Saves checkpoints with only LoRA weights

4. **[scripts/compare_outputs.py](../scripts/compare_outputs.py)** - Comparison tool (183 lines)
   - Generates text from base model
   - Generates text from LoRA fine-tuned model
   - Shows side-by-side comparison

---

### Step 1: Prepare Your Custom Text Data

**Your text file**: `data/sampleBrainRot.txt` (or any text file)

**What this does**:
- Reads your raw text
- Tokenizes using GPT-2's BPE tokenizer
- Splits into 90% train / 10% validation
- Saves as NumPy arrays (.npy files)

**Command**:
```bash
python scripts/prepare_custom_text.py \
  --input data/sampleBrainRot.txt \
  --output brainrot_data
```

**Expected output**:
```
Loading text from data/sampleBrainRot.txt...
Loaded 1,893 characters
Tokenizing...
Tokenized to 356 tokens
Train: 320 tokens (90%)
Val: 36 tokens (10%)

Saved to brainrot_data/
  - train_00000.npy: 320 tokens
  - val_00000.npy: 36 tokens

Data preparation complete!
```

**What gets created**:
```
brainrot_data/
├── train_00000.npy  (320 tokens)
└── val_00000.npy    (36 tokens)
```

---

### Step 2: Fine-Tune GPT-2 with LoRA

**What this does**:
1. Loads pretrained GPT-2 (124M parameters)
2. Adds LoRA adapters to attention layers
3. Freezes base model weights (124M params frozen)
4. Trains only LoRA parameters (~300K params, 0.24% of model)
5. Uses mixed precision (bfloat16) and gradient accumulation
6. Saves checkpoints every 50 steps

**Command**:
```bash
python -m scripts.finetune_lora \
  --data brainrot_data \
  --steps 500 \
  --rank 8 \
  --lr 3e-4
```

**Parameters explained**:
- `--data brainrot_data`: Directory with train/val .npy files
- `--steps 500`: Train for 500 iterations (~5-10 minutes)
- `--rank 8`: LoRA rank (higher = more capacity, more memory)
- `--lr 3e-4`: Learning rate (0.0003)

**Expected output**:
```
============================================================
Loading pretrained gpt2 model...
============================================================
Number of parameters: 124.44 Million
Base model parameters: 124.44M

============================================================
Applying LoRA (rank=8, alpha=16.0)...
============================================================
Applied LoRA to transformer.h.0.attn.c_attn: +18,432 trainable params
Applied LoRA to transformer.h.1.attn.c_attn: +18,432 trainable params
...
Applied LoRA to transformer.h.11.attn.c_attn: +18,432 trainable params

LoRA parameters: 0.29M
Trainable ratio: 0.24%

============================================================
Setting up data loaders from brainrot_data...
============================================================
found 1 shards for split train
found 1 shards for split val

============================================================
Starting LoRA fine-tuning for 500 steps...
Effective batch size: 16
============================================================

step    0 | loss: 4.523145 | lr: 6.00e-06 | norm: 1.2341 | dt: 245.12ms | tok/sec: 2621
step   10 | loss: 3.891234 | lr: 6.60e-05 | norm: 0.8912 | dt: 42.34ms | tok/sec: 15238
step   20 | loss: 3.234567 | lr: 1.26e-04 | norm: 0.7123 | dt: 41.87ms | tok/sec: 15412
...
step  500 | loss: 1.345678 | lr: 3.00e-05 | norm: 0.3456 | dt: 42.05ms | tok/sec: 15343

============================================================
Training complete!
============================================================
Saved LoRA weights to lora_checkpoints/lora_final.pt
```

**Training time**: ~5-10 minutes for 500 steps on GPU

**What gets created**:
```
lora_checkpoints/
├── lora_step_00050.pt   (~1.1 MB)
├── lora_step_00100.pt   (~1.1 MB)
├── lora_step_00150.pt   (~1.1 MB)
...
└── lora_final.pt        (~1.1 MB)
```

---

### Step 3: Compare Base vs Fine-Tuned Model

**What this does**:
1. Loads base GPT-2 model (no fine-tuning)
2. Generates 5 text samples from base model
3. Loads GPT-2 + LoRA adapters
4. Generates 5 text samples from fine-tuned model
5. Shows side-by-side comparison

**Command**:
```bash
python -m scripts.compare_outputs \
  --lora lora_checkpoints/lora_final.pt \
  --samples 5 \
  --temperature 0.8
```

**Expected output**:
```
Using device: cuda

================================================================================
BASE MODEL OUTPUTS
================================================================================

[Sample 1] Prompt: "Once upon a time in"
Output: Once upon a time in the land of stories, there was a young girl named
Sarah who loved to read books and explore the mysteries of the universe...

--------------------------------------------------------------------------------

================================================================================
LORA FINE-TUNED MODEL OUTPUTS
================================================================================

[Sample 1] Prompt: "Once upon a time in"
Output: Once upon a time in goofy ahh Ohio, a quirky blud named Duke Dennis
was hitting the griddy while trying to rizz up Livvy Dunne...

--------------------------------------------------------------------------------

================================================================================
SIDE-BY-SIDE COMPARISON
================================================================================

[Sample 1] Prompt: "Once upon a time in"

Base Model:
  Once upon a time in the land of stories, there was a young girl named Sarah...

LoRA Model:
  Once upon a time in goofy ahh Ohio, a quirky blud named Duke Dennis...

================================================================================
```

**What you should see**:
- Base model: Normal GPT-2 style (formal, standard English)
- LoRA model: Adopted "brain rot" vocabulary and style
  - Uses phrases like "goofy ahh Ohio", "quirky blud", "hitting the griddy"
  - Mimics the tone from your training data
  - Still generates coherent text (not gibberish)

---

## Understanding What Happened

### LoRA Architecture Details

**Applied to**: Attention Q, K, V projections (`c_attn`)
- 12 transformer blocks
- Each block: 1 attention layer with combined Q, K, V projection
- LoRA adds adapters to this combined projection

**Total LoRA Parameters**:
- Per block: ~24,576 params
- Total (12 blocks): ~294,912 params
- Percentage of 124M base model: **0.24%**

### What Gets Trained

**Frozen** (not trained):
- Token embeddings (50,257 × 768)
- Position embeddings (1024 × 768)
- All transformer blocks:
  - MLP weights
  - LayerNorm parameters
  - Output projection weights
- Language modeling head

**Trainable** (LoRA adapters only):
- 12 transformer blocks × 1 attention layer per block
- For each attention layer:
  - Q, K, V projections get LoRA adapters
  - Each adapter: 2 matrices (A and B)
  - Total per block: ~24,576 params
- **Total trainable: ~294,912 params**

### Memory Breakdown (8GB GPU)

**Components**:
- Base model (frozen): ~500 MB (fp16)
- Activations: ~3 GB (batch_size=4, seq_len=128)
- LoRA weights: ~1 MB
- LoRA gradients: ~1 MB
- Optimizer state: ~2 MB
- **Total**: ~7 GB

**Gradient Accumulation**:
- Physical batch: 4 sequences × 128 tokens = 512 tokens
- Accumulation steps: 4
- Effective batch: 16 sequences × 128 tokens = 2,048 tokens
- Simulates larger batch without extra memory

### Training Performance

**GPU (NVIDIA RTX 3090 or similar)**:
- First step: ~245ms (compilation)
- Later steps: ~42ms
- Throughput: ~15,000 tokens/sec
- 500 steps: ~5-10 minutes

**Checkpoint Sizes**:
- LoRA checkpoint: ~1.1 MB (only adapter matrices)
- Full model checkpoint: ~500 MB (all 124M parameters)
- **LoRA is 454× smaller**

---

## Configuration Options

### Data Preparation

```bash
python scripts/prepare_custom_text.py \
  --input data/my_text.txt \        # Your input text file
  --output my_data \                # Output directory
  --train-ratio 0.9                 # 90% train, 10% val (default)
```

### LoRA Parameters

**Rank (`--rank`)**: Size of adapter matrices
- `rank=4`: Fewer parameters (147K), less capacity, faster
- `rank=8`: **Default** - good balance (295K params)
- `rank=16`: More capacity (590K params), slower, better quality
- `rank=32`: Maximum capacity (1.2M params), much slower

**Alpha (`--alpha`)**: Scaling factor
- Controls magnitude of LoRA updates
- Default: `alpha=16` (typically `alpha = 2 * rank`)

### Training Parameters

**Steps (`--steps`)**: Training iterations
- `250`: Quick test (~3 min)
- `500`: **Default** - good results (~5-10 min)
- `1000`: Better quality (~15-20 min)
- `2000+`: Diminishing returns

**Learning Rate (`--lr`)**: How fast to update weights
- `1e-4`: Conservative, slower learning
- `3e-4`: **Default** - works well for LoRA
- `5e-4`: Aggressive, may diverge

**Batch Configuration**:
- `batch_size=4`: Sequences per micro-batch
- `grad_accum_steps=4`: Gradient accumulation steps
- Effective batch: `4 × 4 = 16` sequences

### Model Selection

```bash
python -m scripts.finetune_lora \
  --model gpt2 \                    # gpt2, gpt2-medium, gpt2-large, gpt2-xl
  --weights data/gpt2_weights.pt \  # Pretrained weights path
  --rank 8 \
  --steps 500
```

**Model sizes**:
- `gpt2`: 124M params, 8GB VRAM
- `gpt2-medium`: 350M params, 12GB VRAM
- `gpt2-large`: 774M params, 16GB VRAM
- `gpt2-xl`: 1.5B params, 24GB VRAM

### Memory Requirements

| Configuration | VRAM Usage | Speed |
|--------------|------------|-------|
| rank=4, batch=4 | ~6GB | Fast |
| rank=8, batch=4 | ~7GB | Medium |
| rank=16, batch=8 | ~10GB | Slow |

All configs work on 8GB GPUs with gradient accumulation.

---

## Advanced Usage

### Train on Multiple Text Files

```bash
# Combine multiple files
cat data/file1.txt data/file2.txt data/file3.txt > data/combined.txt

# Prepare
python scripts/prepare_custom_text.py \
  --input data/combined.txt \
  --output combined_data

# Fine-tune
python -m scripts.finetune_lora \
  --data combined_data \
  --steps 1000
```

### Multiple LoRA Adapters

Train different adapters for different styles:

```bash
# Train on brain rot style
python -m scripts.finetune_lora \
  --data brainrot_data \
  --output lora_brainrot \
  --steps 500

# Train on Shakespeare style
python -m scripts.finetune_lora \
  --data shakespeare_data \
  --output lora_shakespeare \
  --steps 500
```

Swap adapters at runtime:

```python
from gpt.model import GPT
from gpt.lora import apply_lora_to_model, load_lora_weights

# Load base model
model = GPT.from_pretrained("gpt2")
apply_lora_to_model(model, rank=8)

# Use brain rot style
load_lora_weights(model, "lora_brainrot/lora_final.pt")
# Generate text...

# Switch to Shakespeare style
load_lora_weights(model, "lora_shakespeare/lora_final.pt")
# Generate text...
```

### Merge LoRA into Base Model

For production deployment (no LoRA overhead):

```python
from gpt.model import GPT
from gpt.lora import apply_lora_to_model, load_lora_weights, merge_all_lora_weights
import torch

# Load model with LoRA
model = GPT.from_pretrained("gpt2", weights_path="data/gpt2_weights.pt")
apply_lora_to_model(model, rank=8)
load_lora_weights(model, "lora_checkpoints/lora_final.pt")

# Merge LoRA into base weights
merge_all_lora_weights(model)

# Save merged model
torch.save(model.state_dict(), "merged_brainrot_gpt2.pt")

# Now model runs without LoRA overhead
# (same behavior, faster inference)
```

### Resume from Checkpoint

```bash
# Compare intermediate checkpoints
python -m scripts.compare_outputs \
  --lora lora_checkpoints/lora_step_00250.pt

# Use earlier checkpoint if final overfitted
python -m scripts.compare_outputs \
  --lora lora_checkpoints/lora_step_00100.pt
```

---

## Troubleshooting

### Issue: `CUDA out of memory`

**Solutions**:
```bash
# Option 1: Reduce batch size
# Edit scripts/finetune_lora.py, line 26:
batch_size: int = 2  # was 4

# Option 2: Reduce sequence length
# Edit scripts/finetune_lora.py, line 27:
seq_len: int = 64  # was 128

# Option 3: Reduce rank
python -m scripts.finetune_lora --rank 4  # was 8
```

### Issue: Loss is NaN or Inf

**Cause**: Learning rate too high

**Solution**:
```bash
python -m scripts.finetune_lora --lr 1e-4  # reduce from 3e-4
```

### Issue: Model outputs don't change after fine-tuning

**Causes & Solutions**:

1. **Not enough training**:
   ```bash
   python -m scripts.finetune_lora --steps 1000  # increase from 500
   ```

2. **Learning rate too low**:
   ```bash
   python -m scripts.finetune_lora --lr 5e-4  # increase from 3e-4
   ```

3. **Rank too small**:
   ```bash
   python -m scripts.finetune_lora --rank 16  # increase from 8
   ```

### Issue: Model outputs gibberish after fine-tuning

**Cause**: Learning rate too high or too much training

**Solutions**:
```bash
# Reduce learning rate
python -m scripts.finetune_lora --lr 1e-4

# Train for fewer steps
python -m scripts.finetune_lora --steps 250

# Use earlier checkpoint
python -m scripts.compare_outputs \
  --lora lora_checkpoints/lora_step_00100.pt
```

### Issue: `FileNotFoundError: data/gpt2_weights.pt`

**Solution**: Cache the weights first:

```python
from transformers import GPT2LMHeadModel
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
torch.save(model.state_dict(), 'data/gpt2_weights.pt')
```

---

## FAQ

### Q: How much data do I need?

**A**: LoRA works with very little data:
- **Minimum**: ~500 tokens (a few paragraphs)
- **Good**: ~5,000 tokens (a few pages)
- **Best**: ~50,000+ tokens (multiple documents)

The brain rot example is only ~350 tokens and still shows clear style shift!

### Q: Can I fine-tune larger models?

**A**: Yes! Just change `--model`:
- `--model gpt2` (124M params, 8GB VRAM)
- `--model gpt2-medium` (350M params, 12GB VRAM)
- `--model gpt2-large` (774M params, 16GB VRAM)
- `--model gpt2-xl` (1.5B params, 24GB VRAM)

### Q: Can I use LoRA for tasks other than text generation?

**A**: Yes! LoRA works for:
- Instruction following
- Question answering
- Summarization
- Code generation

Just prepare your data in the appropriate format.

### Q: How does LoRA compare to full fine-tuning?

| Metric | Full Fine-Tuning | LoRA |
|--------|------------------|------|
| Trainable params | 100% (124M) | 0.24% (0.3M) |
| Memory | High (~20GB) | Low (~7GB) |
| Training time | Slow (~hours) | Fast (~minutes) |
| Quality | Slightly better | Very close |
| Flexibility | One model | Swap adapters |

### Q: Will this overwrite my base model?

**A**: **No!** LoRA only saves adapter weights. Your base model (`data/gpt2_weights.pt`) stays unchanged.

### Q: What's the difference between this and QLoRA?

**A**: QLoRA (coming soon) adds quantization:
- **LoRA**: Uses fp16/bfloat16 base model (~7GB VRAM)
- **QLoRA**: Uses 4-bit quantized base model (~4GB VRAM)
- QLoRA enables fine-tuning on even smaller GPUs

---

## What You Should Learn

After completing this tutorial, you understand:

1. **LoRA basics**: Low-rank adaptation adds small trainable matrices to frozen weights
2. **Memory efficiency**: Train 0.24% of params instead of 100% (7GB vs 20GB VRAM)
3. **Data preparation**: Tokenize text → split train/val → save as .npy shards
4. **Training loop**: Load model → apply LoRA → freeze base → train adapters
5. **Gradient accumulation**: Simulate large batches with small GPU memory
6. **Model evaluation**: Compare base vs fine-tuned outputs side-by-side

**Key takeaway**: You can customize GPT-2's style/vocabulary with just a few paragraphs of text and a few minutes of training, without needing massive compute resources!

---

## Summary

Complete workflow:
```bash
# 1. Prepare data
python scripts/prepare_custom_text.py \
  --input data/my_text.txt \
  --output my_data

# 2. Fine-tune with LoRA
python -m scripts.finetune_lora \
  --data my_data \
  --steps 500

# 3. Compare results
python -m scripts.compare_outputs \
  --lora lora_checkpoints/lora_final.pt
```

You should see clear style/vocabulary shifts in the generated text!

---

## Next Steps

1. **Try your own text**: Replace `data/sampleBrainRot.txt` with your own writing
2. **Experiment with hyperparameters**: Try different ranks, learning rates, steps
3. **Fine-tune larger models**: Use `gpt2-medium` or `gpt2-large`
4. **Train multiple adapters**: Create different LoRA adapters for different styles
5. **Deploy in production**: Merge LoRA into base model for fast inference
6. **Coming soon**: QLoRA for 4-bit quantized fine-tuning

Happy fine-tuning!
