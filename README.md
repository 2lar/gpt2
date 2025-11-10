# GPT-2 Reproduction Workspace

Ensure that you also set up the local virtual environment properly, too.
1. in the workspace run `python3 -m venv env`
2. then to start the virtual environment run `source env/bin/activate`
3. then you can run `pip install -r requirements.txt`

If you keep `data/gpt2_weights.pt` in place, both commands default to it automatically.

## Project Structure

```
gpt2/
├── gpt2/                      # Core model package
│   ├── __init__.py           # Package exports
│   ├── config.py             # Model configuration
│   ├── attention.py          # Multi-head self-attention
│   ├── block.py              # Transformer decoder block
│   ├── model.py              # Complete GPT-2 model
│   ├── data.py               # Data loading utilities
│   └── evaluation.py         # Evaluation helpers
├── scripts/                   # Executable scripts
│   ├── train.py              # Training script
│   ├── generate.py           # Text generation
│   ├── validate.py           # Model validation
│   └── eval_hellaswag.py     # HellaSwag benchmark
├── examples/                  # Reference implementations
│   └── karpathy_original.py  # Original single-file version
├── data/                      # Data directory
│   └── gpt2_weights.pt       # Cached pretrained weights
├── notebooks/                 # Jupyter notebooks for exploration
└── requirements.txt           # Python dependencies
```

## Features

✨ **Educational Focus**: Extensively commented code explaining transformer concepts
📦 **Modular Design**: Clean separation of concerns for easy understanding
🔧 **Production Ready**: Supports distributed training, mixed precision, Flash Attention
📊 **Evaluation**: HellaSwag benchmark and validation against HuggingFace models
🎯 **Weight Loading**: Load pretrained GPT-2 weights from HuggingFace
🎨 **LoRA Fine-Tuning**: Efficient fine-tuning on custom text with 8GB VRAM

## Installation

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Validate Against HuggingFace GPT-2

Compare the implementation against the official GPT-2:

```bash
python scripts/validate.py \
  --model gpt2 \
  --weights data/gpt2_weights.pt \
  --prompt "The quick brown fox" \
  --gen_prompt "Hello, my name is" \
  --max_new_tokens 50
```

This will verify:
- ✅ State dict alignment (weights match after transposition)
- ✅ Logits match (forward pass produces identical outputs)
- ✅ Greedy generation produces same text

### 2. Generate Text

```bash
# Greedy decoding (deterministic)
python scripts/generate.py \
  --source pretrained \
  --model gpt2 \
  --weights data/gpt2_weights.pt \
  --prompt "The three laws of robotics are:" \
  --max_new_tokens 50 \
  --greedy

# Sampling with temperature and nucleus sampling
python scripts/generate.py \
  --source pretrained \
  --model gpt2 \
  --prompt "Once upon a time" \
  --max_new_tokens 100 \
  --temperature 0.8 \
  --top_p 0.95
```

### 3. Train from Scratch

First, prepare a dataset:

```bash
# Quick test dataset (Shakespeare, ~1MB)
python prepare_shakespeare.py
```

Then run training:

```bash
# Single GPU training
python -m scripts.train

# Multi-GPU distributed training
torchrun --standalone --nproc_per_node=8 scripts/train.py
```

See [docs/train.md](docs/train.md) for complete training instructions.

### 4. Fine-Tune on Your Own Text (LoRA)

Quick and efficient fine-tuning using LoRA:

```bash
# Prepare your custom text
python scripts/prepare_custom_text.py \
  --input data/your_text.txt \
  --output custom_data

# Fine-tune with LoRA (5-10 minutes, 8GB VRAM)
python -m scripts.finetune_lora \
  --data custom_data \
  --steps 500

# Compare before/after
python -m scripts.compare_outputs \
  --lora lora_checkpoints/lora_final.pt
```

See [docs/lora_finetuning.md](docs/lora_finetuning.md) for the complete fine-tuning guide.

## Architecture Overview

### GPT-2 Small (124M parameters)

```python
GPTConfig(
    block_size=1024,    # Maximum sequence length
    vocab_size=50257,   # 50k BPE merges + 256 bytes + 1 special token
    n_layer=12,         # Transformer blocks
    n_head=12,          # Attention heads per block
    n_embd=768,         # Embedding dimension
)
```

### Model Sizes

| Model | Layers | Heads | Embedding | Parameters |
|-------|--------|-------|-----------|------------|
| GPT-2 small | 12 | 12 | 768 | 124M |
| GPT-2 medium | 24 | 16 | 1024 | 350M |
| GPT-2 large | 36 | 20 | 1280 | 774M |
| GPT-2 XL | 48 | 25 | 1600 | 1558M |

## Key Concepts Explained in Code

The codebase includes detailed comments explaining:

### Attention (`gpt2/attention.py`)
- Scaled dot-product attention mechanism
- Multi-head parallelization
- Causal masking for autoregressive generation
- Flash Attention optimization

### Transformer Block (`gpt2/block.py`)
- Pre-norm architecture (LayerNorm before sub-layers)
- Residual connections for gradient flow
- MLP with 4x expansion
- Communication (attention) vs. computation (MLP) paradigm

### Model (`gpt2/model.py`)
- Token + position embeddings
- Weight tying between input/output embeddings
- Scaled initialization for deep networks
- Optimizer configuration with selective weight decay

### Training (`scripts/train.py`)
- Distributed data parallel (DDP)
- Gradient accumulation
- Mixed precision (bfloat16)
- Cosine learning rate schedule with warmup
- HellaSwag evaluation
- Checkpointing and logging

## Data Preparation

Prepare the Shakespeare test dataset:

```bash
python prepare_shakespeare.py
```

This downloads tiny Shakespeare (~1MB text), tokenizes it, and creates training shards in `shakespeare_data/`.

For custom datasets, see [docs/train.md](docs/train.md).

## Understanding the Code

Start learning in this order:

1. **`gpt2/config.py`** - Model hyperparameters
2. **`gpt2/attention.py`** - Core attention mechanism
3. **`gpt2/block.py`** - How blocks combine attention + MLP
4. **`gpt2/model.py`** - Full model assembly
5. **`scripts/train.py`** - Complete training loop

Each file contains extensive inline comments explaining:
- **What** the code does
- **Why** specific choices were made
- **How** it relates to the GPT-2 paper

## Performance Optimizations

- ✅ Flash Attention (via `F.scaled_dot_product_attention`)
- ✅ Fused AdamW optimizer
- ✅ Mixed precision training (bfloat16)
- ✅ Gradient accumulation
- ✅ Efficient data loading with shards
- ✅ Distributed training (DDP)

## Weights and Caching

To cache HuggingFace weights locally (avoids repeated downloads):

```python
from transformers import GPT2LMHeadModel
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
torch.save(model.state_dict(), 'data/gpt2_weights.pt')
```

All scripts default to `data/gpt2_weights.pt` if available.

## Resources

- [Andrej Karpathy's video series](https://github.com/karpathy/build-nanogpt)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

## License

Educational purposes. Based on Andrej Karpathy's nanogpt series.
