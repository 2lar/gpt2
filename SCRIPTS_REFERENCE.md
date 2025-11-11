# Scripts Reference

Detailed reference for all scripts with actual usage examples, file paths, and common issues.

---

## Training Scripts (`scripts/training/`)

### Train from Scratch (Modular - Production)

**Command:**
```bash
python -m scripts.training.train
```

**What it does:**
- Trains GPT-2 from scratch using FineWeb-Edu dataset
- Modular version with separated utilities in `lib/`
- Supports multi-GPU training with DDP

**Configuration:**
- Edit `scripts/training/lib/config.py` to change hyperparameters
- Default: 524K tokens/batch, 19073 steps (~10B tokens total)

**Outputs:**
- Checkpoints: `log/model_XXXXX.pt` (every 5000 steps)
- Logs: `log/log.txt`

**Common issues:**
- Requires 16GB+ VRAM for default batch size
- Reduce `micro_batch_size` in config if OOM

---

### Train from Scratch (One-File - Educational)

**Command:**
```bash
python -m scripts.training.onefile_train
```

**What it does:**
- Same as modular version but everything in one file (~500 lines)
- Best for learning and understanding the complete training flow

**Difference:**
- Produces identical results to modular version
- All utilities inlined (no imports from `lib/`)

**📚 See [docs/training_architecture.md](docs/training_architecture.md) for detailed comparison**

---

### LoRA Fine-Tuning

**Command:**
```bash
# Basic usage (with defaults)
python -m scripts.training.finetune_lora

# Custom data and steps
python -m scripts.training.finetune_lora \
  --data brainrot_data \
  --steps 500 \
  --rank 8 \
  --lr 3e-4

# Full options
python -m scripts.training.finetune_lora \
  --data brainrot_data \
  --model gpt2 \
  --weights data/gpt2_weights.pt \
  --steps 500 \
  --rank 8 \
  --lr 3e-4 \
  --output lora_checkpoints
```

**Arguments:**
- `--data`: Data directory (default: `brainrot_data`)
- `--model`: Model size (default: `gpt2`, options: gpt2-medium, gpt2-large, gpt2-xl)
- `--weights`: Path to pretrained weights (default: `data/gpt2_weights.pt`)
- `--steps`: Training steps (default: `500`)
- `--rank`: LoRA rank (default: `8`, higher = more params)
- `--lr`: Learning rate (default: `3e-4`)
- `--output`: Output directory (default: `lora_checkpoints`)

**What it does:**
- Fine-tunes pretrained GPT-2 using LoRA adapters
- Only trains ~0.3M parameters (vs 124M full model)
- Works on 8GB VRAM

**Outputs:**
- Checkpoints: `lora_checkpoints/lora_step_XXXXX.pt` (every 50 steps)
- Final weights: `lora_checkpoints/lora_final.pt`

**Common issues:**
- **Weights path**: If you have weights in `data/weights/gpt2_weights.pt`, use `--weights data/weights/gpt2_weights.pt`
- **Missing data**: First run `prepare_custom_text.py` to create tokenized data

---

## Data Preparation (`scripts/data/`)

### Prepare Custom Text

**Command:**
```bash
# Basic usage
python -m scripts.data.prepare_custom_text \
  --input data/text.txt \
  --output custom_data

# With train/val split ratio
python -m scripts.data.prepare_custom_text \
  --input data/sampleBrainRot.txt \
  --output brainrot_data \
  --train-ratio 0.9
```

**Arguments:**
- `--input`: Path to raw text file (required)
- `--output`: Output directory name (required)
- `--train-ratio`: Train/val split (default: `0.9` = 90% train, 10% val)

**What it does:**
1. Loads raw text file
2. Tokenizes using GPT-2 tokenizer (tiktoken)
3. Splits into train/val
4. Saves as `.npy` shards in output directory

**Output structure:**
```
brainrot_data/
├── train_00000.npy    # Training tokens
└── val_00000.npy      # Validation tokens
```

**Use this output with:**
- `finetune_lora.py --data brainrot_data`

---

### Prepare Shakespeare

**Command:**
```bash
python -m scripts.data.prepare_shakespeare
```

**Arguments:** None (fully automatic)

**What it does:**
- Downloads Shakespeare complete works from web
- Tokenizes and splits into train/val
- Saves to `shakespeare_data/`

**Output:**
```
shakespeare_data/
├── train_00000.npy    # ~1M tokens
└── val_00000.npy      # ~100k tokens
```

**Use case:**
- Quick test dataset for learning
- Validates data pipeline works

---

### Validate Model

**Command:**
```bash
# Basic validation with defaults
python -m scripts.data.validate

# Custom model and weights path
python -m scripts.data.validate \
  --model gpt2 \
  --weights data/weights/gpt2_weights.pt

# Test specific prompts
python -m scripts.data.validate \
  --model gpt2-medium \
  --weights data/gpt2_medium_weights.pt \
  --prompt "Hello world" \
  --gen_prompt "Once upon a time" \
  --max_new_tokens 100
```

**Arguments:**
- `--model`: HuggingFace model size (default: `gpt2`, options: gpt2-medium, gpt2-large, gpt2-xl)
- `--weights`: Path to weights file (default: `data/gpt2_weights.pt`)
- `--prompt`: Prompt for forward/logit test (default: `"The quick brown fox"`)
- `--gen_prompt`: Prompt for generation comparison (default: long prompt)
- `--max_new_tokens`: Tokens to generate (default: `70`)

**What it does:**
1. Loads your GPT implementation
2. Loads HuggingFace reference model
3. Compares outputs to verify correctness

**Common issues:**
- **Weights path**: Update `--weights` to match your folder structure
  - If weights are in `data/weights/`, use `--weights data/weights/gpt2_weights.pt`
  - If weights are in `data/`, use `--weights data/gpt2_weights.pt`

---

## Evaluation (`scripts/evaluation/`)

### HellaSwag Benchmark

**Command:**
```bash
python -m scripts.evaluation.eval_hellaswag
```

**What it does:**
- Downloads HellaSwag validation set (10042 examples)
- Evaluates GPT-2 on common-sense reasoning
- Reports accuracy

**Output:**
- Prints accuracy (GPT-2: ~29%, random: 25%)

---

### Compare Outputs (Before/After LoRA)

**Command:**
```bash
# Basic usage
python -m scripts.evaluation.compare_outputs \
  --lora lora_checkpoints/lora_final.pt

# Full options
python -m scripts.evaluation.compare_outputs \
  --model gpt2 \
  --weights data/gpt2_weights.pt \
  --lora lora_checkpoints/lora_final.pt \
  --rank 8 \
  --alpha 16.0 \
  --samples 5 \
  --max-tokens 50 \
  --temperature 0.8
```

**Arguments:**
- `--lora`: Path to LoRA weights (REQUIRED)
- `--model`: Base model size (default: `gpt2`)
- `--weights`: Base model weights (default: `data/gpt2_weights.pt`)
- `--rank`: LoRA rank (default: `8`, must match training)
- `--alpha`: LoRA alpha (default: `16.0`, must match training)
- `--samples`: Number of samples to generate (default: `5`)
- `--max-tokens`: Max tokens per sample (default: `50`)
- `--temperature`: Sampling temperature (default: `0.8`)

**What it does:**
1. Generates text with base model (before LoRA)
2. Generates text with LoRA-adapted model (after LoRA)
3. Shows side-by-side comparison

**Common issues:**
- **Rank mismatch**: `--rank` must match what you used during training
- **Weights path**: Update `--weights` if your weights are in `data/weights/`

---

## Inference (`scripts/inference/`)

### Generate Text

**Command:**
```bash
# Use pretrained weights (default)
python -m scripts.inference.generate \
  --source pretrained \
  --model gpt2 \
  --prompt "Once upon a time" \
  --max_new_tokens 100

# Use custom weights in nested folder
python -m scripts.inference.generate \
  --source pretrained \
  --model gpt2 \
  --weights data/weights/gpt2_weights.pt \
  --prompt "The meaning of life is" \
  --max_new_tokens 50

# Use local checkpoint from training
python -m scripts.inference.generate \
  --source local \
  --model gpt2 \
  --ckpt log/model_19000.pt \
  --prompt "Hello world" \
  --max_new_tokens 100

# Greedy decoding (deterministic)
python -m scripts.inference.generate \
  --source pretrained \
  --model gpt2 \
  --prompt "To be or not to be" \
  --greedy

# Custom sampling parameters
python -m scripts.inference.generate \
  --source pretrained \
  --model gpt2 \
  --prompt "AI will" \
  --max_new_tokens 100 \
  --temperature 0.9 \
  --top_k 50 \
  --top_p 0.95
```

**Arguments:**

**Source (required):**
- `--source`: `pretrained` or `local`
  - `pretrained`: Load from HuggingFace-format weights (`.pt` file with state dict)
  - `local`: Load from training checkpoint (`.pt` file with `model`, `config`, `step`, etc.)

**Model:**
- `--model`: Model size (default: `gpt2`, options: gpt2-medium, gpt2-large, gpt2-xl)

**Weights (depends on source):**
- `--weights`: Path to HF-format weights when `--source=pretrained` (default: `data/gpt2_weights.pt`)
  - **IMPORTANT**: Update this if your weights are in a subfolder like `data/weights/`
  - Example: `--weights data/weights/gpt2_weights.pt`
- `--ckpt`: Path to training checkpoint when `--source=local` (e.g., `log/model_19000.pt`)

**Generation:**
- `--prompt`: Text prompt (default: `"The three laws of robotics are:"`)
- `--max_new_tokens`: Number of tokens to generate (default: `50`)

**Sampling:**
- `--greedy`: Use greedy decoding (deterministic, picks highest probability)
- `--temperature`: Sampling temperature (default: `0.9`, higher = more random)
- `--top_k`: Top-k sampling (default: `None`, example: `50`)
- `--top_p`: Nucleus sampling (default: `0.95`, range: 0-1)

**What it does:**
- Loads model from pretrained weights or training checkpoint
- Generates text from a prompt
- Uses temperature/top-k/top-p sampling or greedy decoding

**Common issues:**
- **"FileNotFoundError"**: Check `--weights` path matches your folder structure
  - If weights are in `data/weights/gpt2_weights.pt`, use `--weights data/weights/gpt2_weights.pt`
  - If weights are in `data/gpt2_weights.pt`, use `--weights data/gpt2_weights.pt` (default)
- **"KeyError: 'model'"**: You used `--source=local` but pointed to HF weights
  - Use `--source=pretrained` for `.pt` files from `download_hf_weights.py`
  - Use `--source=local` only for checkpoints saved during training

---

### Generate Text with LoRA

**Command:**
```bash
# Basic usage
python -m scripts.inference.generate_lora \
  --lora lora_checkpoints/lora_final.pt \
  --prompt "Once upon a time"

# With custom weights path
python -m scripts.inference.generate_lora \
  --weights data/weights/gpt2_weights.pt \
  --lora lora_checkpoints/lora_final.pt \
  --prompt "Your custom prompt here" \
  --max_new_tokens 200

# With different LoRA checkpoint
python -m scripts.inference.generate_lora \
  --lora lora_checkpoints/lora_step_00500.pt \
  --prompt "Hello world" \
  --max_new_tokens 100

# Greedy decoding
python -m scripts.inference.generate_lora \
  --lora lora_checkpoints/lora_final.pt \
  --prompt "To be or not to be" \
  --greedy

# Custom sampling parameters
python -m scripts.inference.generate_lora \
  --lora lora_checkpoints/lora_final.pt \
  --prompt "In the future" \
  --max_new_tokens 150 \
  --temperature 0.8 \
  --top_k 40 \
  --top_p 0.9
```

**Arguments:**

**LoRA Configuration (REQUIRED):**
- `--lora`: Path to LoRA weights checkpoint (REQUIRED)
- `--rank`: LoRA rank (default: `8`, **must match training**)
- `--alpha`: LoRA alpha (default: `16.0`, **must match training**)

**Model:**
- `--model`: Base model size (default: `gpt2`, options: gpt2-medium, gpt2-large, gpt2-xl)
- `--weights`: Path to base model weights (default: `data/gpt2_weights.pt`)

**Generation:**
- `--prompt`: Text prompt (default: `"Once upon a time"`)
- `--max_new_tokens`: Number of tokens to generate (default: `100`)

**Sampling:**
- `--greedy`: Use greedy decoding (deterministic)
- `--temperature`: Sampling temperature (default: `0.9`, higher = more random)
- `--top_k`: Top-k sampling (default: `None`, example: `40`)
- `--top_p`: Nucleus sampling (default: `0.95`, range: 0-1)

**What it does:**
1. Loads base GPT-2 model
2. Applies LoRA adapters with specified rank/alpha
3. Loads fine-tuned LoRA weights
4. Generates text using the adapted model

**Common issues:**
- **Rank mismatch**: `--rank` and `--alpha` must match what you used during training
  - Check your training command or [finetune_lora.py](scripts/training/finetune_lora.py) defaults
  - Default training uses rank=8, alpha=16.0
- **Weights path**: If weights are in `data/weights/`, use `--weights data/weights/gpt2_weights.pt`
- **FileNotFoundError**: Check both `--weights` and `--lora` paths are correct

---

## Running Scripts Directly (Alternative)

You can also run scripts directly without the `-m` flag:

```bash
# These are equivalent
python -m scripts.training.train
python scripts/training/train.py

# These are equivalent
python -m scripts.data.prepare_custom_text --input data/text.txt --output custom_data
python scripts/data/prepare_custom_text.py --input data/text.txt --output custom_data
```

**When to use `-m` flag:**
- ✅ Recommended: More consistent with Python module conventions
- ✅ Required if script imports from sibling packages

**When to run directly:**
- Quick iteration and testing
- Simpler for copy-pasting commands

---

## Common File Paths

Based on typical project structure:

```
gpt2/
├── data/
│   ├── gpt2_weights.pt              # HF weights (if in root of data/)
│   └── weights/
│       └── gpt2_weights.pt          # HF weights (if in subfolder)
├── brainrot_data/
│   ├── train_00000.npy              # Custom training data
│   └── val_00000.npy                # Custom validation data
├── shakespeare_data/
│   ├── train_00000.npy              # Shakespeare train
│   └── val_00000.npy                # Shakespeare val
├── lora_checkpoints/
│   ├── lora_step_00050.pt           # LoRA checkpoints
│   └── lora_final.pt                # LoRA final weights
└── log/
    ├── model_05000.pt               # Training checkpoints
    ├── model_19000.pt               # Final checkpoint
    └── log.txt                      # Training logs
```

**Adjusting commands for your structure:**

If your weights are in `data/weights/gpt2_weights.pt`, add to ALL commands:
```bash
--weights data/weights/gpt2_weights.pt
```

Examples:
```bash
# Generate with nested weights
python -m scripts.inference.generate \
  --weights data/weights/gpt2_weights.pt \
  --prompt "Hello"

# Validate with nested weights
python -m scripts.data.validate \
  --weights data/weights/gpt2_weights.pt

# LoRA fine-tune with nested weights
python -m scripts.training.finetune_lora \
  --weights data/weights/gpt2_weights.pt \
  --data brainrot_data
```

---

## Quick Start Workflow

### 1. Validate your GPT-2 implementation
```bash
python -m scripts.data.validate
```

### 2. Prepare custom data
```bash
python -m scripts.data.prepare_custom_text \
  --input data/sampleBrainRot.txt \
  --output brainrot_data
```

### 3. Fine-tune with LoRA
```bash
python -m scripts.training.finetune_lora \
  --data brainrot_data \
  --steps 500
```

### 4. Compare before/after
```bash
python -m scripts.evaluation.compare_outputs \
  --lora lora_checkpoints/lora_final.pt
```

### 5. Generate text with LoRA model
```bash
python -m scripts.inference.generate_lora \
  --lora lora_checkpoints/lora_final.pt \
  --prompt "Your prompt here" \
  --max_new_tokens 100
```

### 6. (Optional) Generate with base model for comparison
```bash
python -m scripts.inference.generate \
  --prompt "Your prompt here" \
  --max_new_tokens 100
```

---

## Module Import Style (For Developers)

When writing scripts, use absolute imports from project root:

```python
from gpt.model import GPT
from gpt.config import GPTConfig
from gpt.data import DataLoaderLite
from gpt.lora import apply_lora_to_model
```

This works because the project root is in `PYTHONPATH` when running scripts.
