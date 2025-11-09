# Training Guide

Complete instructions for training the GPT-2 model from scratch.

---

## Prerequisites

1. **Python Environment Setup**

   From the project root:

   ```bash
   # Create virtual environment (one-time setup)
   python3 -m venv env

   # Activate virtual environment (required every time)
   source env/bin/activate

   # Install dependencies (one-time setup)
   pip install -r requirements.txt
   ```

2. **Verify Installation**

   ```bash
   python -c "from gpt.model import GPT; print('Success!')"
   ```

---

## Quick Start

### Step 1: Prepare Dataset

```bash
python prepare_shakespeare.py
```

**Output:**
```
Saved 304,222 train tokens and 33,803 val tokens
```

Creates `shakespeare_data/` with tokenized training data.

### Step 2: Run Training

```bash
python -m scripts.train
```

**Important:** Use `python -m scripts.train` (not `python scripts/train.py`)
- The `-m` flag ensures imports work correctly
- Without it: `ModuleNotFoundError: No module named 'gpt'`

**Expected Output:**
```
found 1 shards for split train
found 1 shards for split val
using device: cuda
step 0 | loss: 10.8234 | lr: 0.000006 | dt: 245ms
step 1 | loss: 9.2341 | lr: 0.000012 | dt: 42ms
step 50 | loss: 4.5123 | lr: 0.000300 | dt: 41ms
val loss 4.4523
```

Training takes ~5 minutes for 1000 steps on GPU.

Press `Ctrl+C` to stop. Checkpoints saved to `log/`.

---

## Understanding Training

### Data Flow

```
Text → Tokenizer → .npy shards → DataLoader → Model → Loss → Optimizer
```

### Key Hyperparameters

From [scripts/train.py](../scripts/train.py):

```python
data_root: str = "shakespeare_data"  # Dataset directory
total_batch_size: int = 32768        # Total tokens per batch
micro_batch_size: int = 16           # Batch size per GPU
seq_len: int = 256                   # Sequence length
max_lr: float = 6e-4                 # Peak learning rate
warmup_steps: int = 100              # LR warmup steps
max_steps: int = 1000                # Total training steps
```

**Batch Sizes:**
- `total_batch_size`: Tokens before weight update
- `micro_batch_size`: Tokens per forward pass
- Gradient accumulation handles the difference

**Learning Rate:**
- Warmup: Linear increase to `max_lr`
- Decay: Cosine decay to `max_lr * 0.1`

### Model Architecture

GPT-2 Small (124M parameters):
```python
block_size=1024    # Max sequence length
vocab_size=50257   # Tokenizer vocabulary
n_layer=12         # Transformer blocks
n_head=12          # Attention heads
n_embd=768         # Embedding dimension
```

---

## Checkpoints

### Structure After Training

```
log/
├── model_00000.pt
├── model_00050.pt
├── model_00100.pt
└── ...
```

### Loading a Checkpoint

```python
from gpt.model import GPT
from gpt.config import GPTConfig
import torch

checkpoint = torch.load('log/model_01000.pt')
config = GPTConfig(**checkpoint['config'])
model = GPT(config)
model.load_state_dict(checkpoint['model'])
```

---

## Monitoring

### Key Metrics

**Training Loss:**
- Starts ~10, drops to ~1.5-2.0
- Should decrease steadily

**Validation Loss:**
- Should track training loss
- If val >> train → overfitting

**Gradient Norm:**
- Typical: 0.5-5.0
- If >10 → potential instability

**Time per Step:**
- First step: slow (compilation)
- Later steps: 40-100ms on GPU

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'gpt'`**
- Use `python -m scripts.train` (not `python scripts/train.py`)

**`FileNotFoundError: shakespeare_data`**
- Run `python prepare_shakespeare.py` first

**`CUDA out of memory`**
- Reduce `micro_batch_size` from 16 to 8 or 4

**Loss is NaN/Inf:**
- Reduce learning rate from 6e-4 to 3e-4

**Training very slow:**
- Check GPU with `nvidia-smi`
- Verify output shows `using device: cuda`

---

## Multi-GPU Training

```bash
# 4 GPUs
torchrun --standalone --nproc_per_node=4 scripts/train.py

# 8 GPUs
torchrun --standalone --nproc_per_node=8 scripts/train.py
```

Gradient accumulation automatically adjusts for faster training.

---

## Custom Datasets

### From Text Files

```python
import numpy as np
import tiktoken
from pathlib import Path

# Load text
texts = [open(f, 'r').read() for f in ["file1.txt", "file2.txt"]]
all_text = "\n".join(texts)

# Tokenize
enc = tiktoken.get_encoding("gpt2")
tokens = np.array(enc.encode(all_text), dtype=np.int32)

# Split 90/10
split = int(len(tokens) * 0.9)

# Save
output = Path("my_data")
output.mkdir(exist_ok=True)
np.save(output / "train_00000.npy", tokens[:split])
np.save(output / "val_00000.npy", tokens[split:])
```

Update `data_root` in [scripts/train.py](../scripts/train.py) to `"my_data"`.

### From HuggingFace

```python
from datasets import load_dataset
import tiktoken
import numpy as np

# Load dataset
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")

# Tokenize
enc = tiktoken.get_encoding("gpt2")
tokens = []
for ex in ds:
    tokens.extend(enc.encode(ex["text"]))

tokens = np.array(tokens, dtype=np.int32)
split = int(len(tokens) * 0.9)

# Save
np.save("wikitext_data/train_00000.npy", tokens[:split])
np.save("wikitext_data/val_00000.npy", tokens[split:])
```

---

## After Training

**Generate text:**
```bash
python -m scripts.generate --checkpoint log/model_01000.pt --prompt "Once upon a time"
```

**Evaluate:**
```bash
python -m scripts.eval_hellaswag --checkpoint log/model_01000.pt
```

**Validate:**
```bash
python -m scripts.validate --model gpt2 --prompt "Hello world"
```

---

## FAQ

**Q: How long does training take?**

A: ~5 minutes for 1000 steps on GPU

**Q: Can I resume training?**

A: Yes, automatically loads latest checkpoint from `log/`

**Q: What GPU do I need?**

A: Minimum 8GB VRAM. Recommended 16GB+.

**Q: Can I train on CPU?**

A: Yes, but 100x slower than GPU

---

## Summary

```bash
# Setup (once)
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Prepare data
python prepare_shakespeare.py

# Train
python -m scripts.train
```

---

**Happy Training!**
