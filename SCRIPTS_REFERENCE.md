# Scripts Reference

Quick reference for all scripts in the reorganized structure.

## Training Scripts (`scripts/training/`)

### Train from Scratch (Modular Version - Recommended for Production)
```bash
python -m scripts.training.train
```
Professional modular training with separated utilities.
- Clean, maintainable code
- Reusable components
- Best practices demonstrated

### Train from Scratch (One-File Version - Best for Learning)
```bash
python -m scripts.training.onefile_train
```
Complete training in one file for educational purposes.
- Everything in one place
- Easy to understand flow
- Great for learning

**📚 See [docs/training_architecture.md](docs/training_architecture.md) for detailed comparison**

### LoRA Fine-Tuning
```bash
python -m scripts.training.finetune_lora --data brainrot_data --steps 500
```
Fine-tune a pretrained model using LoRA adapters.

## Data Preparation (`scripts/data/`)

### Prepare Custom Text
```bash
python -m scripts.data.prepare_custom_text --input data/text.txt --output custom_data
```
Convert raw text files into tokenized .npy shards.

### Prepare Shakespeare
```bash
python -m scripts.data.prepare_shakespeare
```
Download and prepare Shakespeare dataset (quick test dataset).

### Validate Model
```bash
python -m scripts.data.validate --model gpt2 --weights data/gpt2_weights.pt
```
Validate implementation against HuggingFace GPT-2.

## Evaluation (`scripts/evaluation/`)

### HellaSwag Benchmark
```bash
python -m scripts.evaluation.eval_hellaswag
```
Run HellaSwag common-sense reasoning benchmark.

### Compare Outputs
```bash
python -m scripts.evaluation.compare_outputs --lora lora_checkpoints/lora_final.pt
```
Compare text generation before/after LoRA fine-tuning.

## Inference (`scripts/inference/`)

### Generate Text
```bash
python -m scripts.inference.generate \
  --source pretrained \
  --model gpt2 \
  --prompt "Once upon a time" \
  --max_new_tokens 100
```
Generate text from a trained model.

## Running Scripts Directly

You can also run scripts directly (for quick iteration):

```bash
# Works without -m flag
python scripts/training/train.py
python scripts/data/prepare_custom_text.py --input data/text.txt --output custom_data
```

## Module Import Style

For imports within scripts, use absolute imports from project root:

```python
from gpt.model import GPT
from gpt.config import GPTConfig
from gpt.data import DataLoaderLite
```
