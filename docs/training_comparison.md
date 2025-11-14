# Training Strategy Comparison Guide

Quick reference to help you choose the right training approach for your use case.

---

## Overview of Training Scripts

Your workspace now has **7 different training scripts** for different scenarios:

| Script | Use Case | GPUs | Memory Efficiency | Speed |
|--------|----------|------|-------------------|-------|
| `onefile_train.py` | **Learning/Education** | 1 or DDP | Normal | Fast |
| `train.py` (modular) | **Production from-scratch** | 1 or DDP | Normal | Fast |
| `train_fsdp.py` | **Large-scale distributed** | 2+ (FSDP2) | **Excellent** | Good |
| `finetune_lora.py` | **Quick fine-tuning** | 1 | Good | **Fastest** |
| `finetune_qlora.py` | **Memory-constrained fine-tuning** | 1 | **Excellent** | Good |

---

## Decision Tree: Which Script Should I Use?

```
START: What do you want to do?
│
├─ Train from scratch (no pretrained weights)?
│  │
│  ├─ Learning how training works?
│  │  └─ Use: onefile_train.py
│  │     - Everything in one file
│  │     - Easy to understand
│  │     - ~500 lines of self-contained code
│  │
│  ├─ Production training (1 GPU or small model)?
│  │  └─ Use: train.py (modular)
│  │     - Clean, separated utilities
│  │     - Supports DDP for multi-GPU
│  │     - Professional code structure
│  │
│  └─ Large model that doesn't fit on 1 GPU?
│     └─ Use: train_fsdp.py
│        - FSDP2 sharding across GPUs
│        - 4-8x memory reduction
│        - Requires 2+ GPUs
│
└─ Fine-tune existing model (with pretrained weights)?
   │
   ├─ Have plenty of VRAM (8GB+)?
   │  └─ Use: finetune_lora.py
   │     - Regular LoRA (fast)
   │     - 0.5% trainable params
   │     - Works on 8GB VRAM
   │
   └─ Limited VRAM (4GB) or want to fine-tune larger models?
      └─ Use: finetune_qlora.py
         - Quantized LoRA (memory-efficient)
         - 8x memory reduction on base model
         - Works on 4GB VRAM
```

---

## Detailed Comparison

### 1. Training from Scratch

#### onefile_train.py vs train.py (modular)

**onefile_train.py** (Educational):
```python
✅ All code in one file (~500 lines)
✅ Easy to read top-to-bottom
✅ Great for learning
✅ Self-contained (no lib/ imports)
❌ Less maintainable for production
❌ Code duplication if you need variants

Use when: Learning, understanding training, teaching others
```

**train.py** (Production):
```python
✅ Clean separation of concerns
✅ Reusable utilities in lib/
✅ Easy to maintain and extend
✅ Supports DDP multi-GPU
❌ More files to navigate
❌ Requires understanding module structure

Use when: Production training, team projects, need to extend
```

**Both produce identical results!** The only difference is code organization.

#### train.py (DDP) vs train_fsdp.py (FSDP2)

**train.py with DDP** (Multi-GPU, replicated):
```python
Memory per GPU:
├─ Parameters: 496 MB (full model on each GPU)
├─ Gradients: 496 MB
├─ Optimizer: 992 MB
└─ Total: ~2 GB per GPU

✅ Simpler to use (just set CUDA_VISIBLE_DEVICES)
✅ Faster for models that fit on 1 GPU
✅ Better for debugging
❌ Each GPU needs full model
❌ Doesn't scale to very large models

Use when: Model fits on 1 GPU, have 4-8 GPUs, want simple multi-GPU
Command: torchrun --nproc_per_node=4 scripts/training/train.py
```

**train_fsdp.py** (FSDP2, sharded):
```python
Memory per GPU (4 GPUs):
├─ Parameters: 124 MB (sharded 1/4)
├─ Gradients: 124 MB (sharded 1/4)
├─ Optimizer: 248 MB (sharded 1/4)
└─ Total: ~500 MB per GPU

✅ 4x memory reduction (can train 4x larger model)
✅ Scales to 100+ GPUs
✅ Essential for models > 1B params
❌ More complex (sharding, DTensor)
❌ Slight overhead from communication
❌ Harder to debug

Use when: Model doesn't fit on 1 GPU, want max efficiency, 2+ GPUs
Command: torchrun --nproc_per_node=4 scripts/training/train_fsdp.py
```

**Example: GPT-2 1.5B parameters**
- DDP: Needs 6GB per GPU (1.5B × 4 bytes) just for params → **Won't fit on 8GB GPUs**
- FSDP2 (4 GPUs): Needs 1.5GB per GPU (1.5B / 4 × 4 bytes) → **Fits easily!**

---

### 2. Fine-Tuning

#### finetune_lora.py vs finetune_qlora.py

**finetune_lora.py** (Regular LoRA):
```python
Memory breakdown (GPT-2 124M):
├─ Base model (fp32): 496 MB (frozen)
├─ LoRA adapters (fp32): 0.6 MB (trainable)
├─ Gradients: 0.6 MB (only for LoRA)
├─ Optimizer: 1.2 MB (only for LoRA)
├─ Activations: ~6 GB (depends on batch size)
└─ Total: ~7 GB peak

✅ Faster training (no dequantization)
✅ Faster inference
✅ Simpler implementation
❌ Needs 8GB+ VRAM
❌ Can't fine-tune larger models on consumer GPUs

Use when: Have 8GB+ VRAM, want speed, GPT-2/small models
Command: python -m scripts.training.finetune_lora --data your_data
```

**finetune_qlora.py** (Quantized LoRA):
```python
Memory breakdown (GPT-2 124M):
├─ Base model (4-bit NF4): 62 MB (frozen, quantized)
├─ LoRA adapters (bf16): 0.6 MB (trainable)
├─ Gradients: 0.6 MB
├─ Optimizer: 1.2 MB
├─ Activations: ~6 GB (depends on batch size)
└─ Total: ~6.5 GB peak

✅ 8x memory reduction on base model
✅ Can fine-tune on 4GB GPUs
✅ Can fine-tune larger models (GPT-2 medium, large)
❌ 10-20% slower (dequantization overhead)
❌ Slower inference

Use when: Limited VRAM (4-6GB), want to fine-tune larger models
Command: python -m scripts.training.finetune_qlora --data your_data
```

**Example: Fine-tuning GPT-2 Medium (350M params)**
- Regular LoRA: 350M × 4 bytes = 1.4 GB base model → **Needs 16GB+ VRAM total**
- QLoRA: 350M × 0.5 bytes = 175 MB base model → **Works on 8GB VRAM!**

---

## Memory Requirements Summary

### Training from Scratch (GPT-2 124M)

| Setup | Params Memory | Optimizer Memory | Peak VRAM | Min GPU |
|-------|---------------|------------------|-----------|---------|
| Single GPU | 496 MB | 992 MB | ~8 GB | 8 GB |
| DDP (4 GPUs) | 496 MB/GPU | 992 MB/GPU | ~8 GB/GPU | 8 GB |
| FSDP2 (4 GPUs, fp32) | 124 MB/GPU | 248 MB/GPU | ~6.5 GB/GPU | 8 GB |
| FSDP2 (4 GPUs, bf16) | 62 MB/GPU | 248 MB/GPU | ~3.5 GB/GPU | 4 GB |

### Fine-Tuning (GPT-2 124M)

| Setup | Base Model | Trainable | Peak VRAM | Min GPU |
|-------|------------|-----------|-----------|---------|
| Full fine-tuning | 496 MB | 496 MB | ~8 GB | 8 GB |
| LoRA | 496 MB | 0.6 MB | ~7 GB | 8 GB |
| QLoRA | 62 MB | 0.6 MB | ~6.5 GB | 4 GB |

---

## Speed Comparison

Relative training speed (higher is faster):

| Method | Speed | Throughput | Notes |
|--------|-------|------------|-------|
| Single GPU | 1.0x | 1000 tok/s | Baseline |
| DDP (4 GPUs) | 3.8x | 3800 tok/s | Near-linear scaling |
| FSDP2 (4 GPUs) | 3.2x | 3200 tok/s | Communication overhead |
| LoRA | 1.2x | 1200 tok/s | Faster (fewer params) |
| QLoRA | 0.9x | 900 tok/s | Dequantization overhead |

**Note**: These are approximate and depend on hardware, model size, and batch size.

---

## When to Use What: Common Scenarios

### Scenario 1: Learning PyTorch / GPT-2

**Recommendation**: `onefile_train.py`

Why?
- Everything in one file, easy to read
- Understand the full training pipeline
- Modify and experiment easily

### Scenario 2: Training GPT-2 124M from Scratch (1 GPU)

**Recommendation**: `train.py`

Why?
- Clean, production-ready code
- Can switch to DDP later if needed
- Well-structured for extension

### Scenario 3: Training GPT-2 124M from Scratch (4 GPUs)

**Recommendation**: `train.py` with DDP

Why?
- Model fits on 1 GPU, so DDP is simpler
- Near-linear scaling
- Less communication overhead than FSDP2

Command:
```bash
torchrun --nproc_per_node=4 scripts/training/train.py
```

### Scenario 4: Training GPT-2 1.5B from Scratch (4 × 8GB GPUs)

**Recommendation**: `train_fsdp.py`

Why?
- 1.5B params won't fit on 8GB GPU with DDP
- FSDP2 shards model across GPUs
- Enables training larger models

Command:
```bash
torchrun --nproc_per_node=4 scripts/training/train_fsdp.py \
  --n-layer 48 --n-head 16 --n-embd 1600
```

### Scenario 5: Fine-tune GPT-2 on Custom Data (1 × 8GB GPU)

**Recommendation**: `finetune_lora.py`

Why?
- Fast and simple
- 8GB is enough for LoRA
- Good quality results

Command:
```bash
python -m scripts.training.finetune_lora \
  --data your_data \
  --steps 500
```

### Scenario 6: Fine-tune GPT-2 on Custom Data (1 × 4GB GPU)

**Recommendation**: `finetune_qlora.py`

Why?
- 4GB isn't enough for regular LoRA
- QLoRA reduces memory 8x
- Still good quality

Command:
```bash
python -m scripts.training.finetune_qlora \
  --data your_data \
  --steps 500
```

### Scenario 7: Fine-tune GPT-2 Medium on Custom Data (1 × 8GB GPU)

**Recommendation**: `finetune_qlora.py`

Why?
- GPT-2 Medium (350M) won't fit in 8GB with regular LoRA
- QLoRA: 350M × 0.5 bytes = 175 MB base model
- Enables fine-tuning larger models on consumer GPUs

Command:
```bash
python -m scripts.training.finetune_qlora \
  --model gpt2-medium \
  --data your_data \
  --steps 1000
```

---

## Summary Table

| Your Goal | Your Hardware | **Recommended Script** |
|-----------|---------------|------------------------|
| Learn training | 1 GPU | `onefile_train.py` |
| Production training (small model) | 1 GPU | `train.py` |
| Production training (small model) | 4 GPUs | `train.py` + DDP |
| Train large model (1B+) | 4+ GPUs | `train_fsdp.py` |
| Fine-tune (8GB VRAM) | 1 GPU | `finetune_lora.py` |
| Fine-tune (4GB VRAM) | 1 GPU | `finetune_qlora.py` |
| Fine-tune large model | 1 GPU | `finetune_qlora.py` |

---

## Quick Command Reference

```bash
# Learning / Education
python -m scripts.training.onefile_train

# Production training (single GPU)
python -m scripts.training.train

# Production training (multi-GPU DDP)
torchrun --nproc_per_node=4 scripts/training/train

# Large-scale distributed (FSDP2)
torchrun --nproc_per_node=4 scripts/training/train_fsdp

# Fine-tuning (regular LoRA)
python -m scripts.training.finetune_lora --data your_data

# Fine-tuning (memory-efficient QLoRA)
python -m scripts.training.finetune_qlora --data your_data
```

---

## Further Reading

- **[docs/training_architecture.md](training_architecture.md)** - onefile vs modular training
- **[docs/fsdp2_explained.md](fsdp2_explained.md)** - Deep dive into FSDP2
- **[docs/qlora_explained.md](qlora_explained.md)** - Deep dive into QLoRA
- **[docs/qlora_code_walkthrough.md](qlora_code_walkthrough.md)** - QLoRA implementation details
- **[SCRIPTS_REFERENCE.md](../SCRIPTS_REFERENCE.md)** - All scripts with examples

---

**In summary**: You have the complete toolkit for any GPT-2 training scenario! 🚀
