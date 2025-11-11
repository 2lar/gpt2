# Training Architecture: Two Versions

This project provides two training scripts to serve different learning purposes:

## 1. One-File Training (`onefile_train.py`)

**Purpose**: Educational - understand the complete training loop in one place

**File**: `scripts/training/onefile_train.py`

**Best for**:
- Learning how training works end-to-end
- Understanding the complete flow
- Following Andrej Karpathy's GPT-2 reproduction style
- Debugging (everything in one place)

**Structure**:
```python
# Single file (~500 lines)
- TrainingConfig
- setup_distributed()
- get_lr()
- evaluate_loss()
- build_dataloaders()
- train()  # Main training loop
```

**Run it**:
```bash
python -m scripts.training.onefile_train
```

## 2. Modular Training (`train.py`)

**Purpose**: Professional - demonstrate best practices and code organization

**Files**:
```
scripts/training/
├── train.py                  # Main entry point (~250 lines)
└── lib/                      # Reusable utilities
    ├── __init__.py           # Public API
    ├── config.py             # TrainingConfig (shared)
    ├── distributed.py        # DDP setup/teardown
    ├── scheduling.py         # Learning rate schedules
    ├── evaluation.py         # Eval loops & sampling
    └── data.py               # Data loader builders
```

**Best for**:
- Production code
- Reusing components across projects
- Team collaboration (work on different modules)
- Testing individual components
- Scaling to larger projects

**Run it**:
```bash
python -m scripts.training.train
```

---

## Comparison

| Aspect | onefile_train.py | train.py (modular) |
|--------|------------------|---------------------|
| **Lines of code** | ~500 | ~250 + utilities |
| **Readability** | Linear, top-to-bottom | Requires jumping between files |
| **Reusability** | Copy-paste | Import utilities |
| **Testability** | Test whole script | Test individual components |
| **Debugging** | Everything in one place | Need to trace imports |
| **Configuration** | Inline dataclass | Shared config.py |
| **Best for** | Learning | Production |
| **Maintainability** | Good for small projects | Better for large projects |

---

## Module Breakdown (Modular Version)

### `lib/config.py`
Shared training configuration:
- `TrainingConfig` - Dataclass with all hyperparameters
- Used by both train.py and other scripts

**Why separate?**
- Single source of truth for config
- DRY principle (avoid duplication)
- Easy to modify defaults in one place
- Can be imported by other training scripts

### `lib/distributed.py`
Handles multi-GPU training setup:
- `setup_distributed()` - Initialize DDP, detect devices
- `teardown_distributed()` - Clean up resources
- Uses modern `torch.accelerator` API with fallback

**Why separate?**
- Reusable across train/eval/inference scripts
- Complex logic with device detection
- Independent of training loop

### `lib/scheduling.py`
Learning rate scheduling strategies:
- `get_lr()` - Warmup + cosine decay schedule

**Why separate?**
- Easy to add new schedules (linear, exponential, etc.)
- Testable in isolation
- Math-heavy code separated from training logic

### `lib/evaluation.py`
Model evaluation during training:
- `evaluate_loss()` - Validation loss computation
- `run_hellaswag_eval()` - Benchmark evaluation
- `generate_samples()` - Text generation

**Why separate?**
- Different evaluation strategies
- Reusable for standalone evaluation scripts
- Can be tested with mock models

### `lib/data.py`
Data loading utilities:
- `build_dataloaders()` - Create train/val loaders

**Why separate?**
- DRY principle (used in train and eval)
- Easy to add new data sources
- Configuration logic centralized

---

## Which Should You Use?

### Use `onefile_train.py` when:
- ✅ You're learning PyTorch training
- ✅ You want to understand the complete flow
- ✅ You're debugging a specific issue
- ✅ You're following the "Reproduce GPT-2" tutorial
- ✅ You prefer simple, linear code

### Use `train.py` (modular) when:
- ✅ You're building a production system
- ✅ You want to reuse components
- ✅ You're working in a team
- ✅ You need to test individual parts
- ✅ You're scaling to multiple training scripts

---

## Learning Path Recommendation

1. **Start with `onefile_train.py`**
   - Read it top to bottom
   - Understand each section
   - Run it and observe the flow

2. **Then explore `train.py` (modular)**
   - See how it's refactored
   - Understand separation of concerns
   - Learn professional patterns

3. **Compare both versions**
   - Notice what stayed the same (training loop logic)
   - Notice what was extracted (utilities)
   - Understand the trade-offs

---

## Code Organization Principles (Modular Version)

### Single Responsibility Principle
Each module has one job:
- `distributed.py` → DDP setup
- `scheduling.py` → LR scheduling
- `evaluation.py` → Model evaluation
- `data.py` → Data loading

### DRY (Don't Repeat Yourself)
Utilities are imported, not copied:
```python
# Instead of copy-pasting setup_distributed() into every script:
from .lib import setup_distributed
```

### Testability
Each function can be tested independently:
```python
# Test LR schedule without running full training
from scripts.training.lib import get_lr
assert get_lr(step=0, cfg=mock_config) == 0.0
```

### Separation of Concerns
Training loop focuses on orchestration:
```python
# Main loop is clean and readable
val_loss = evaluate_loss(model, val_loader, ...)
lr = get_lr(step, cfg)
optimizer.step()
```

---

## Advanced: Creating Your Own Utilities

The modular structure makes it easy to add new components:

### Example: Add a new LR schedule

1. Add to `lib/scheduling.py`:
```python
def get_lr_linear(step: int, cfg: ScheduleConfig) -> float:
    """Linear decay schedule."""
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    decay = 1.0 - (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.max_lr * max(decay, cfg.min_lr_ratio)
```

2. Export in `lib/__init__.py`:
```python
from .scheduling import get_lr, get_lr_linear
```

3. Use in `train.py`:
```python
from .lib import get_lr_linear
# ...
lr = get_lr_linear(step, train_cfg)
```

---

## Summary

Both versions teach valuable lessons:

**`onefile_train.py`** teaches:
- How training works end-to-end
- The complete training loop
- PyTorch fundamentals

**`train.py` (modular)** teaches:
- Code organization
- Separation of concerns
- Professional software engineering

**Use both** to become a well-rounded ML engineer!
