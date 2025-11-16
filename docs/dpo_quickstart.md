# DPO Quick Start Guide

Complete guide to training GPT-2 with DPO (Direct Preference Optimization) from scratch.

---

## What You Have

A **complete, from-scratch DPO implementation** with:
- ✅ Pure PyTorch (no HuggingFace TRL library)
- ✅ ~600 lines of heavily commented educational code
- ✅ Complete algorithm explanation
- ✅ Sample data generator
- ✅ Ready to run!

**Files created:**
- [scripts/training/train_dpo.py](../scripts/training/train_dpo.py) - Main DPO training script
- [scripts/data/create_preference_data.py](../scripts/data/create_preference_data.py) - Data creation helper
- [docs/dpo_explained.md](dpo_explained.md) - Deep dive into DPO theory

---

## Quick Start (3 Steps)

### Step 1: Create Preference Data

```bash
# Generate sample preference data
python -m scripts.data.create_preference_data --output preference_data.jsonl
```

This creates a JSONL file with 10 sample preference pairs like:
```json
{"prompt": "Explain machine learning", "chosen": "Detailed answer...", "rejected": "Short answer"}
```

### Step 2: Train with DPO

```bash
# Train on the preference data
python -m scripts.training.train_dpo \
  --data preference_data.jsonl \
  --steps 1000
```

**What happens:**
1. Loads pretrained GPT-2
2. Creates frozen reference copy
3. Trains to prefer "chosen" over "rejected" responses
4. Saves aligned model to `dpo_checkpoints/`

### Step 3: Use the Aligned Model

```bash
# Generate text with the aligned model
python -m scripts.inference.generate \
  --weights dpo_checkpoints/dpo_final.pt \
  --prompt "Explain quantum computing"
```

The model should now give more detailed, helpful responses!

---

## Understanding the Data Format

DPO requires preference pairs in JSONL format:

```json
{
  "prompt": "The input question or instruction",
  "chosen": "The better, preferred response (detailed, helpful)",
  "rejected": "The worse response (short, low-quality)"
}
```

### Example 1: Helpfulness

```json
{
  "prompt": "What is photosynthesis?",
  "chosen": "Photosynthesis is the process by which plants convert light energy into chemical energy. Plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. This occurs in chloroplasts containing chlorophyll. The equation is: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂.",
  "rejected": "Plants use sunlight to make food."
}
```

### Example 2: Safety

```json
{
  "prompt": "How do I bypass website security?",
  "chosen": "I can't help with bypassing security systems as that would be unethical and potentially illegal. If you're interested in cybersecurity, I'd be happy to discuss legitimate learning resources like Capture The Flag competitions or educational platforms.",
  "rejected": "You can try SQL injection or XSS attacks."
}
```

### Example 3: Format

```json
{
  "prompt": "List three planets.",
  "chosen": "Here are three planets:\n1. Earth - our home planet\n2. Mars - the red planet\n3. Jupiter - the largest planet",
  "rejected": "earth mars jupiter"
}
```

---

## Creating Your Own Preference Data

### Option 1: Interactive Mode

```bash
# Add your own examples interactively
python -m scripts.data.create_preference_data --interactive --output my_prefs.jsonl
```

The script will prompt you for:
1. Prompt
2. Chosen response
3. Rejected response

Type "done" when finished.

### Option 2: From Existing Data

If you have existing data, convert it to JSONL:

```python
import json

preferences = [
    {
        "prompt": "Your prompt",
        "chosen": "Better response",
        "rejected": "Worse response"
    },
    # ... more examples
]

with open('my_preferences.jsonl', 'w') as f:
    for pref in preferences:
        f.write(json.dumps(pref) + '\n')
```

### Option 3: From Model Generations

Generate pairs by:
1. Prompting your model twice
2. Manually labeling which response is better
3. Creating preference pairs

---

## Training Parameters Explained

### Beta (Temperature)

```bash
--beta 0.1  # Default: gentle preference
--beta 0.5  # Higher: stronger preference for chosen
--beta 0.01 # Lower: very gentle (stays closer to reference)
```

**What it does:**
- Controls how strongly to prefer chosen over rejected
- Higher β → model becomes more confident in preferences
- Lower β → model stays closer to original behavior

**Rule of thumb:**
- Start with 0.1
- Increase if model doesn't improve enough
- Decrease if model diverges or becomes overconfident

### Learning Rate

```bash
--lr 5e-7  # Default: very low for stability
--lr 1e-6  # Higher: faster learning (more risky)
--lr 1e-7  # Lower: more stable (slower)
```

**Why so low?**
- Model is already pretrained
- We're fine-tuning, not training from scratch
- DPO is sensitive to learning rate
- Too high → model diverges

**Rule of thumb:**
- Start with 5e-7
- Increase by 2x if training is too slow
- Decrease by 2x if loss increases or diverges

### Batch Size and Gradient Accumulation

```bash
--batch-size 4 --grad-accum 4  # Effective batch = 16
--batch-size 2 --grad-accum 8  # Same effective batch, less VRAM
```

**Effective batch size** = batch_size × grad_accum

**Rule of thumb:**
- Effective batch of 8-32 works well
- Reduce batch_size if OOM
- Increase grad_accum to maintain effective batch size

---

## Monitoring Training

### Key Metrics

DPO training logs several important metrics:

**Loss:**
```
step   100 | loss: 0.693147 | reward_margin: +0.0000 | accuracy: 0.500
step   200 | loss: 0.520000 | reward_margin: +0.3500 | accuracy: 0.750
step   300 | loss: 0.350000 | reward_margin: +0.8000 | accuracy: 0.875
```

**What they mean:**

1. **Loss**: DPO loss (lower is better)
   - Random model: ~0.69 (log(2))
   - Well-trained: 0.1-0.4
   - If increasing: decrease LR or beta

2. **Reward Margin**: Difference in implicit rewards
   - Positive: Model prefers chosen over rejected (good!)
   - Negative: Model prefers rejected (bad!)
   - Higher is better (typically 0.5-2.0 at convergence)

3. **Accuracy**: % of examples where model prefers chosen
   - Random: 50%
   - Well-trained: 80-95%
   - If stuck at 50%: increase beta or LR

### Good Training

```
step     0 | loss: 0.693 | reward_margin: +0.000 | accuracy: 0.500  # Starting
step   100 | loss: 0.520 | reward_margin: +0.350 | accuracy: 0.750  # Learning
step   500 | loss: 0.280 | reward_margin: +1.200 | accuracy: 0.875  # Converging
step  1000 | loss: 0.210 | reward_margin: +1.500 | accuracy: 0.900  # Done!
```

Loss decreases, reward margin increases, accuracy increases.

### Bad Training (Diverging)

```
step     0 | loss: 0.693 | reward_margin: +0.000 | accuracy: 0.500
step   100 | loss: 0.950 | reward_margin: -0.500 | accuracy: 0.300  # Getting worse!
step   200 | loss: 1.850 | reward_margin: -1.200 | accuracy: 0.100  # Diverging!
```

Loss increasing, negative reward margin → **decrease LR and/or beta**

---

## Common Issues and Solutions

### Issue 1: "Preference data not found"

**Error:**
```
FileNotFoundError: Preference data not found at preference_data.jsonl
```

**Solution:**
```bash
# Create sample data first
python -m scripts.data.create_preference_data --output preference_data.jsonl
```

### Issue 2: Loss not decreasing

**Symptoms:**
- Loss stays around 0.69 (log(2))
- Reward margin stays near 0
- Accuracy stays at 50%

**Solutions:**
1. Increase beta: `--beta 0.3` (stronger preference signal)
2. Increase learning rate: `--lr 1e-6`
3. Check data quality (chosen actually better than rejected?)
4. Train longer: `--steps 2000`

### Issue 3: Loss increasing (divergence)

**Symptoms:**
- Loss increases over time
- Reward margin becomes negative
- Accuracy decreases

**Solutions:**
1. Decrease learning rate: `--lr 1e-7`
2. Decrease beta: `--beta 0.05`
3. Increase gradient accumulation: `--grad-accum 8`

### Issue 4: Out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size
--batch-size 2

# Increase gradient accumulation (maintains effective batch size)
--grad-accum 8

# Use smaller model
--model gpt2  # instead of gpt2-medium
```

### Issue 5: Very slow training

**Cause:** Loading full model twice (policy + reference) uses ~1GB VRAM

**Solutions:**
- Use smaller model initially
- Expect ~2x memory vs regular fine-tuning
- This is unavoidable for DPO (needs reference model)

---

## Advanced Usage

### Multi-Objective DPO

Train on multiple preference types:

```python
# Create different preference files
helpful_prefs.jsonl  # Helpfulness preferences
safe_prefs.jsonl     # Safety preferences

# Train on each sequentially
python -m scripts.training.train_dpo --data helpful_prefs.jsonl --steps 500 --output dpo_helpful
python -m scripts.training.train_dpo --weights dpo_helpful/dpo_final.pt --data safe_prefs.jsonl --steps 500 --output dpo_safe
```

### Custom Beta Schedule

Modify `train_dpo.py` to increase beta during training:

```python
# In training loop
beta = config.beta * (1 + step / config.max_steps)  # Linearly increase beta
```

This starts gentle and becomes stronger over time.

### Validation-Based Early Stopping

Monitor validation loss and stop when it increases:

```python
# Add to training loop
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter >= 3:
        print("Early stopping!")
        break
```

---

## Comparison: DPO vs HuggingFace DPOTrainer

### Your Implementation (train_dpo.py)

```python
# ~600 lines, pure PyTorch
from gpt.model import GPT
import torch.nn.functional as F

# You write the loss function
def dpo_loss(policy_logps, ref_logps, beta):
    logits = beta * (policy_logps - ref_logps)
    return -F.logsigmoid(logits).mean()

# You control everything
for batch in dataloader:
    loss, metrics = dpo_loss(...)
    loss.backward()
    optimizer.step()
```

**Pros:**
- ✅ Complete understanding
- ✅ Full control
- ✅ Easy to modify
- ✅ Educational
- ✅ No dependencies

**Cons:**
- ❌ More code to write
- ❌ Need to maintain
- ❌ Less tested than HF

### HuggingFace DPOTrainer

```python
# ~20 lines, high-level API
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    train_dataset=dataset,
    args=config,
)
trainer.train()
```

**Pros:**
- ✅ Very concise
- ✅ Well-tested
- ✅ Maintained by HF
- ✅ Quick to use

**Cons:**
- ❌ Black box
- ❌ Harder to customize
- ❌ Dependencies

**When to use which:**
- Learning: Use your implementation
- Research: Use your implementation
- Production (standard): Use HF
- Production (custom): Use your implementation

---

## Next Steps

### 1. Try It Out

```bash
# Generate sample data
python -m scripts.data.create_preference_data

# Train DPO
python -m scripts.training.train_dpo --data preference_data.jsonl --steps 1000

# Test the model
python -m scripts.inference.generate --weights dpo_checkpoints/dpo_final.pt
```

### 2. Create Real Preference Data

Collect real preferences:
- Have humans rate model outputs
- Compare model outputs and label which is better
- Use existing preference datasets (Anthropic HH-RLHF, etc.)

### 3. Experiment with Parameters

Try different settings:
- Beta: 0.05, 0.1, 0.3, 0.5
- Learning rate: 1e-7, 5e-7, 1e-6
- Model size: gpt2, gpt2-medium

### 4. Combine with Other Techniques

Full pipeline:
```bash
# 1. Pretrain (or use existing model)
python -m scripts.training.train

# 2. Instruction tuning (SFT with LoRA)
python -m scripts.training.finetune_lora --data instructions.txt

# 3. Preference alignment (DPO)
python -m scripts.training.train_dpo --data preferences.jsonl

# Result: Aligned, helpful model!
```

### 5. Read the Code

Understand every line:
1. Read [scripts/training/train_dpo.py](../scripts/training/train_dpo.py)
2. Focus on `dpo_loss()` function (the core algorithm)
3. Understand `compute_sequence_log_probs()` (how we evaluate sequences)
4. Trace through one training step

### 6. Compare to RLHF

Read about traditional RLHF:
- Requires reward model training
- Requires PPO (complex RL algorithm)
- DPO simplifies this massively

---

## Summary

**You now have:**
- ✅ Complete DPO implementation from scratch
- ✅ Sample data generator
- ✅ Comprehensive documentation
- ✅ Training guide with examples
- ✅ Troubleshooting guide

**Key takeaways:**
1. DPO aligns models to preferences without RL
2. Requires: model + reference copy + preference data
3. Loss function: prefer chosen over rejected, stay close to reference
4. Beta controls preference strength
5. Very low learning rate (model already pretrained)

**Recommended learning path:**
1. ✅ Read this quickstart
2. ✅ Run the training with sample data
3. ✅ Read [train_dpo.py](../scripts/training/train_dpo.py) to understand implementation
4. ✅ Read [dpo_explained.md](dpo_explained.md) for deep theory
5. ✅ Create your own preference data and train!

Happy aligning! 🎯
