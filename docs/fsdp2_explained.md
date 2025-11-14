# FSDP2 (Fully Sharded Data Parallel v2) Explained

This document provides a deep dive into FSDP2, how it works, and how to use it effectively.

---

## Table of Contents

1. [What is FSDP2?](#what-is-fsdp2)
2. [Why Use FSDP2?](#why-use-fsdp2)
3. [FSDP2 vs Other Parallelism Strategies](#fsdp2-vs-other-strategies)
4. [How FSDP2 Works](#how-fsdp2-works)
5. [DTensor: The Foundation](#dtensor-the-foundation)
6. [Memory Breakdown](#memory-breakdown)
7. [Forward and Backward Pass](#forward-and-backward-pass)
8. [Best Practices](#best-practices)
9. [Common Issues](#common-issues)
10. [Advanced Topics](#advanced-topics)

---

## What is FSDP2?

**FSDP2 (Fully Sharded Data Parallel version 2)** is PyTorch's modern distributed training framework that enables training large models by sharding (splitting) them across multiple GPUs.

### Key Concepts

**Sharding**: Splitting tensors across devices
- Each GPU holds only a portion (shard) of the model
- Parameters, gradients, and optimizer states are all sharded
- Shards are gathered when needed, then freed

**DTensor**: Distributed Tensor
- New tensor type in PyTorch 2.x
- Represents a tensor that's distributed across devices
- Looks like a regular tensor to your code
- Automatically handles communication

**fully_shard()**: The FSDP2 API
- Function (not a class) that wraps your model
- Converts regular parameters to sharded DTensors
- Much cleaner than FSDP1's wrapper class

---

## Why Use FSDP2?

### Memory Efficiency

FSDP2 reduces memory usage by sharding everything:

```
Single GPU (124M param model):
├─ Parameters: 496 MB (124M × 4 bytes)
├─ Gradients: 496 MB
├─ Optimizer state (AdamW): 992 MB (2 states × 124M × 4 bytes)
└─ TOTAL: ~2 GB just for model + optimizer!

FSDP2 (4 GPUs):
├─ Parameters: 124 MB per GPU (496 MB / 4)
├─ Gradients: 124 MB per GPU
├─ Optimizer state: 248 MB per GPU
└─ TOTAL: ~500 MB per GPU (4x reduction!)
```

### Scale to Large Models

With FSDP2, you can train models that don't fit on a single GPU:

- GPT-3 175B: Would need 700 GB for model alone
- With FSDP2 on 128 GPUs: ~5.5 GB per GPU
- Makes large models accessible!

### Better Than Model Parallelism

Compared to naive model parallelism (splitting layers across GPUs):
- ✅ Better GPU utilization (all GPUs active)
- ✅ No pipeline bubbles
- ✅ Easier to implement
- ✅ Works with any model architecture

---

## FSDP2 vs Other Strategies

### DDP (DistributedDataParallel)

```
DDP:
- Each GPU: Full model copy
- Forward: Independent on each GPU
- Backward: All-reduce gradients
- Memory: N × model size (N = num GPUs)
- Use when: Model fits on 1 GPU, want data parallelism

FSDP2:
- Each GPU: 1/N of model
- Forward: All-gather params, compute, free
- Backward: All-gather params, compute gradients, reduce-scatter
- Memory: 1 × model size (total across all GPUs)
- Use when: Model too large for 1 GPU
```

### FSDP1 (Old API)

```
FSDP1:
- API: FullyShardedDataParallel(model) wrapper class
- Implementation: FlatParameter (flattened buffers)
- Complexity: More complex state management
- Status: Legacy, still works but not recommended

FSDP2:
- API: fully_shard(model) function
- Implementation: DTensor (native distributed tensors)
- Complexity: Cleaner, more composable
- Status: Modern, recommended for new projects
```

### Pipeline Parallelism

```
Pipeline Parallelism:
- Split layers across GPUs sequentially
- GPU 0: Layers 0-2
- GPU 1: Layers 3-5
- GPU 2: Layers 6-8
- Issue: Pipeline bubbles (GPUs idle)

FSDP2:
- All GPUs participate in each layer
- No idle time
- Better utilization
```

### Tensor Parallelism

```
Tensor Parallelism:
- Split individual tensors across GPUs
- Example: Split 8192-dim vector into 4×2048
- Requires model code changes
- Often combined with FSDP2

FSDP2:
- Parameter sharding (splits along batch dimension)
- No model code changes needed
- Can be combined with tensor parallelism (2D parallelism)
```

---

## How FSDP2 Works

### The Sharding Process

When you call `fully_shard(model)`:

```python
# Before FSDP2
model.layer.weight  # torch.Tensor on cuda:0
                   # Shape: [768, 768]
                   # Memory: 768 × 768 × 4 bytes = 2.3 MB on GPU 0

# After FSDP2 (4 GPUs)
model.layer.weight  # DTensor sharded across [cuda:0, cuda:1, cuda:2, cuda:3]
                   # GPU 0: [192, 768] (first 192 rows)
                   # GPU 1: [192, 768] (next 192 rows)
                   # GPU 2: [192, 768] (next 192 rows)
                   # GPU 3: [192, 768] (last 192 rows)
                   # Memory per GPU: 192 × 768 × 4 bytes = 0.58 MB
```

### Sharding Dimensions

FSDP2 typically shards along dimension 0 (rows):

```python
# Original weight matrix
W = [
    [w00, w01, w02, ..., w0n],   # Row 0
    [w10, w11, w12, ..., w1n],   # Row 1
    ...
    [wm0, wm1, wm2, ..., wmn],   # Row m
]

# After sharding across 4 GPUs
GPU 0: rows 0 to m/4
GPU 1: rows m/4 to m/2
GPU 2: rows m/2 to 3m/4
GPU 3: rows 3m/4 to m
```

Why dimension 0?
- Most matrix multiplications: `output = input @ W.T`
- Sharding rows of W allows parallel computation
- Each GPU can compute independently

---

## DTensor: The Foundation

### What is DTensor?

DTensor (Distributed Tensor) is the core abstraction powering FSDP2:

```python
# Regular tensor
x = torch.randn(1024, 768, device='cuda:0')  # Lives on one GPU

# DTensor (sharded)
from torch.distributed._tensor import DTensor
x_sharded = DTensor(...)  # Sharded across multiple GPUs
                          # Each GPU holds a piece
                          # Looks like a full tensor to your code
```

### DTensor Operations

DTensors support all standard PyTorch operations:

```python
# These all work transparently!
y = x_sharded + 1
z = torch.nn.functional.relu(x_sharded)
w = x_sharded @ weight_sharded  # Automatic communication!
```

### Sharding Specs

DTensor has a "sharding spec" that describes how it's distributed:

```python
# Sharded on dimension 0
placement = [Shard(0)]  # Shard along dim 0
# GPU 0: [0:256, :]
# GPU 1: [256:512, :]
# GPU 2: [512:768, :]
# GPU 3: [768:1024, :]

# Replicated (full copy on each GPU)
placement = [Replicate()]
# GPU 0: [0:1024, :]  (full)
# GPU 1: [0:1024, :]  (full)
# ...

# Partial (different values, need reduction)
placement = [Partial()]
# Used for gradients before all-reduce
```

---

## Memory Breakdown

### Without FSDP2 (Single GPU, 124M params)

```
Component               Memory
──────────────────────────────
Parameters (fp32)       496 MB   (124M × 4 bytes)
Gradients (fp32)        496 MB   (124M × 4 bytes)
Optimizer momentum      496 MB   (124M × 4 bytes)
Optimizer variance      496 MB   (124M × 4 bytes)
──────────────────────────────
Subtotal               ~2.0 GB

Activations (depends on batch size)
  batch=16, seq=1024   ~6 GB
──────────────────────────────
TOTAL                  ~8 GB
```

### With FSDP2 (4 GPUs, 124M params)

```
Per GPU:
Component               Memory
──────────────────────────────
Parameters (sharded)    124 MB   (31M × 4 bytes per GPU)
Gradients (sharded)     124 MB
Optimizer momentum      124 MB
Optimizer variance      124 MB
──────────────────────────────
Subtotal               ~500 MB per GPU

Activations (per GPU)
  batch=16, seq=1024   ~6 GB
──────────────────────────────
TOTAL                  ~6.5 GB per GPU

Peak during forward:
  + Gathered params     ~500 MB (temporary)
──────────────────────────────
Peak                   ~7 GB per GPU
```

### With FSDP2 + Mixed Precision (4 GPUs)

```
Per GPU:
Component               Memory
──────────────────────────────
Parameters (bf16, sharded) 62 MB   (31M × 2 bytes)
Gradients (bf16, sharded)  62 MB
Optimizer (fp32, sharded) 248 MB   (31M × 8 bytes for AdamW)
──────────────────────────────
Subtotal               ~370 MB per GPU

Activations (bf16)
  batch=16, seq=1024   ~3 GB
──────────────────────────────
TOTAL                  ~3.4 GB per GPU

Peak during forward:
  + Gathered params     ~250 MB (temporary)
──────────────────────────────
Peak                   ~3.7 GB per GPU
```

**Result: Can train on 4GB GPUs!**

---

## Forward and Backward Pass

### Forward Pass (Detailed)

Here's what happens during `loss = model(x)` with FSDP2:

```
Step 1: Start with sharded parameters
  GPU 0: W0 [192, 768]
  GPU 1: W1 [192, 768]
  GPU 2: W2 [192, 768]
  GPU 3: W3 [192, 768]

Step 2: All-gather parameters (when layer is used)
  All GPUs now have full W = [W0; W1; W2; W3] = [768, 768]
  Communication: all-gather (efficient NCCL operation)

Step 3: Compute forward pass
  Each GPU computes: output = input @ W.T
  (Different inputs per GPU due to data parallelism)

Step 4: Free gathered parameters
  Free the full W, keep only shard
  GPU 0: W0 [192, 768]
  GPU 1: W1 [192, 768]
  ...

Step 5: Repeat for next layer
  Gather → Compute → Free → Gather → Compute → Free → ...

Final: Compute loss
  Each GPU has different loss (different data)
```

### Backward Pass (Detailed)

Here's what happens during `loss.backward()`:

```
Step 1: Start from loss
  Each GPU has its own loss value
  Backward propagates through layers in reverse

Step 2: For each layer (in reverse):
  a) All-gather parameters (same as forward)
     All GPUs get full W = [W0; W1; W2; W3]

  b) Compute gradients
     Each GPU computes: dL/dW based on its input/grad_output
     Result: Each GPU has FULL gradient dL/dW [768, 768]

  c) Reduce-scatter gradients
     Sum gradients across GPUs and split the result:
       GPU 0 gets sum(dL/dW)[0:192, :] (shard 0)
       GPU 1 gets sum(dL/dW)[192:384, :] (shard 1)
       GPU 2 gets sum(dL/dW)[384:576, :] (shard 2)
       GPU 3 gets sum(dL/dW)[576:768, :] (shard 3)
     Communication: reduce-scatter (sum + split)

  d) Free gathered parameters
     Keep only sharded gradient

Step 3: After all layers
  Each GPU has:
    - Shard of parameters
    - Shard of gradients (summed across GPUs)
  Ready for optimizer step!
```

### Communication Primitives

FSDP2 uses two main collective operations:

**All-Gather** (used in forward/backward to gather params):
```
Before:
GPU 0: [A]
GPU 1: [B]
GPU 2: [C]
GPU 3: [D]

After (all GPUs have same):
All GPUs: [A, B, C, D]
```

**Reduce-Scatter** (used in backward to shard gradients):
```
Before:
GPU 0: [A0, B0, C0, D0]
GPU 1: [A1, B1, C1, D1]
GPU 2: [A2, B2, C2, D2]
GPU 3: [A3, B3, C3, D3]

After (summed and split):
GPU 0: [A0+A1+A2+A3]
GPU 1: [B0+B1+B2+B3]
GPU 2: [C0+C1+C2+C3]
GPU 3: [D0+D1+D2+D3]
```

---

## Best Practices

### 1. Sharding Granularity

**Per-layer sharding** (recommended):
```python
# Shard each transformer block separately
for i, block in enumerate(model.transformer.h):
    model.transformer.h[i] = fully_shard(block)

# Then shard root
model = fully_shard(model)
```

Benefits:
- Better communication/computation overlap
- Lower peak memory usage
- More flexible checkpointing

**Whole-model sharding** (simpler):
```python
# One FSDP unit for entire model
model = fully_shard(model)
```

Benefits:
- Simpler code
- Fewer FSDP units to manage

### 2. Mixed Precision

Always use mixed precision with FSDP2:

```python
from torch.distributed.fsdp import MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,   # Store params in bf16
    reduce_dtype=torch.float32,   # Reduce grads in fp32
)

model = fully_shard(model, mixed_precision=mp_policy)
```

Why?
- 2x memory savings on parameters
- Faster computation (bf16 is faster on modern GPUs)
- fp32 gradient reduction maintains stability

### 3. Gradient Accumulation

FSDP2 works perfectly with gradient accumulation:

```python
optimizer.zero_grad()

for micro_step in range(grad_accum_steps):
    loss = model(x) / grad_accum_steps
    loss.backward()  # Gradients accumulate in shards

optimizer.step()  # Each GPU updates its shard
```

No special handling needed - it just works!

### 4. Learning Rate Scaling

With FSDP2, you're doing data parallelism across GPUs:

```python
# If using 4 GPUs, effective batch size is 4x larger
# You may want to scale learning rate:
lr = base_lr * world_size

# Or use gradient accumulation to keep effective batch size same
```

### 5. Logging

Only rank 0 should log to avoid duplicate output:

```python
if rank == 0:
    print(f"Loss: {loss.item()}")
    wandb.log({"loss": loss.item()})
```

But all ranks should participate in collective operations:

```python
# All ranks participate in all_reduce
dist.all_reduce(loss, op=dist.ReduceOp.AVG)

# Then only rank 0 logs
if rank == 0:
    print(f"Loss: {loss.item()}")
```

---

## Common Issues

### Issue 1: "NCCL error: unhandled system error"

**Cause**: GPUs can't communicate (network issue, wrong device)

**Fix**:
```python
# Make sure each process uses different GPU
torch.cuda.set_device(local_rank)

# Check NCCL debug output
os.environ['NCCL_DEBUG'] = 'INFO'
```

### Issue 2: "RuntimeError: Expected all tensors to be on the same device"

**Cause**: Model and data on different devices

**Fix**:
```python
# Move model to correct device BEFORE fully_shard
model = model.to(f'cuda:{local_rank}')
model = fully_shard(model)

# Move data to same device
x = x.to(f'cuda:{local_rank}')
```

### Issue 3: Out of memory even with FSDP2

**Causes**:
- Activations still large (reduce batch size)
- Gradient accumulation not working (check division)
- Too many all-gathers at once (reduce sharding granularity)

**Fixes**:
```python
# 1. Reduce batch size per GPU
batch_size = 4  # Instead of 16

# 2. Use activation checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

for block in model.transformer.h:
    block = checkpoint_wrapper(block)  # Recompute activations in backward

# 3. Use CPU offloading (slower but saves memory)
model = fully_shard(model, cpu_offload=True)
```

### Issue 4: Slow training

**Causes**:
- Too much communication (too fine-grained sharding)
- CPU offloading enabled
- Slow interconnect

**Fixes**:
```python
# 1. Use coarser sharding
model = fully_shard(model)  # One unit instead of per-layer

# 2. Disable CPU offload
model = fully_shard(model, cpu_offload=False)

# 3. Check interconnect (should be NVLink or InfiniBand)
# Check with: nvidia-smi topo -m
```

---

## Advanced Topics

### 1. Hybrid Sharding

For multi-node training, use hybrid sharding:

```python
# Shard within each node, replicate across nodes
# This reduces inter-node communication (which is slower)

# 2D DeviceMesh: [nodes, gpus_per_node]
mesh = DeviceMesh("cuda", [[0, 1, 2, 3], [4, 5, 6, 7]])  # 2 nodes, 4 GPUs each

model = fully_shard(model, mesh=mesh)
```

### 2. Activation Checkpointing

Trade computation for memory:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

# Wrap blocks to recompute activations in backward
for i, block in enumerate(model.transformer.h):
    block = checkpoint_wrapper(block)
    model.transformer.h[i] = fully_shard(block)
```

Memory savings:
- Without: Store all activations for backward pass
- With: Recompute activations during backward (2x compute, 1/N memory)

### 3. Distributed Checkpointing

For very large models, save sharded checkpoints:

```python
import torch.distributed.checkpoint as dist_checkpoint

# Save sharded checkpoint (each rank saves its shard)
dist_checkpoint.save_state_dict(
    state_dict=model.state_dict(),
    storage_writer=dist_checkpoint.FileSystemWriter("checkpoint/"),
)

# Load sharded checkpoint
dist_checkpoint.load_state_dict(
    state_dict=model.state_dict(),
    storage_reader=dist_checkpoint.FileSystemReader("checkpoint/"),
)
```

Benefits:
- Faster save/load (parallel I/O)
- No need to gather to rank 0
- Required for models > 100B parameters

### 4. Composing with Other Parallelism

FSDP2 can be combined with other strategies:

**FSDP + Tensor Parallelism (2D parallelism)**:
```python
# Use DeviceMesh to specify 2D topology
mesh_2d = DeviceMesh("cuda", [[0, 1], [2, 3]])  # 2×2 mesh

# Apply tensor parallelism within rows
# Apply FSDP across columns
```

**FSDP + Pipeline Parallelism**:
```python
# Split model into stages
# Apply FSDP within each stage
```

---

## Summary

**Key Takeaways**:

1. **FSDP2 shards everything**: params, gradients, optimizer states
2. **DTensor is the foundation**: Distributed tensors that act like regular tensors
3. **Communication is automatic**: All-gather in forward, reduce-scatter in backward
4. **Memory savings are massive**: ~4x for model, ~8x for model+optimizer
5. **Use per-layer sharding**: Better performance and flexibility
6. **Always use mixed precision**: 2x additional memory savings
7. **Checkpointing requires gathering**: Use distributed checkpointing for large models

**When to use FSDP2**:
- ✅ Model doesn't fit on 1 GPU
- ✅ Want to train larger models on available hardware
- ✅ Multi-GPU single-node or multi-node setup
- ✅ Need memory efficiency

**When NOT to use FSDP2**:
- ❌ Model easily fits on 1 GPU (use DDP for simplicity)
- ❌ Very small models (overhead not worth it)
- ❌ Debugging (more complex than single-GPU)

**Next steps**:
1. Read [scripts/training/train_fsdp.py](../scripts/training/train_fsdp.py) - heavily commented training script
2. Try training with FSDP2 on your model
3. Monitor memory usage with `nvidia-smi`
4. Experiment with sharding granularity
5. Profile to find bottlenecks

Happy distributed training! 🚀
