"""
train.py: Training entry point that mirrors the reference example while keeping
modules separated.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2.config import GPTConfig
from gpt2.data import DataLoaderLite, LoaderConfig
from gpt2.evaluation import get_most_likely_row
from gpt2.model import GPT

try:
    import tiktoken
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError("tiktoken is required for training") from exc

try:
    # HellaSwag script is in the same directory (scripts/)
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from eval_hellaswag import iterate_examples, render_example
except (ModuleNotFoundError, ImportError):
    iterate_examples = render_example = None


@dataclass
class TrainingConfig:
    data_root: str = "edu_fineweb10B"
    total_batch_size: int = 524288
    micro_batch_size: int = 64
    seq_len: int = 1024
    eval_interval: int = 250
    eval_iters: int = 20
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 715
    max_steps: int = 19073
    log_dir: str = "log"
    compile: bool = False
    hellaswag_interval: int = 250
    sampling_interval: int = 250
    sample_prompt: str = "Hello, I'm a language model,"
    sample_max_len: int = 32
    sample_count: int = 4
    weight_decay: float = 0.1


def setup_distributed() -> Tuple[bool, int, int, int, str, str, bool]:
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP run requires CUDA"
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, str(device), device_type, master_process


def teardown_distributed(enabled: bool) -> None:
    if enabled:
        dist.destroy_process_group()


def get_lr(step: int, cfg: TrainingConfig) -> float:
    min_lr = cfg.max_lr * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    if step > cfg.max_steps:
        return min_lr
    decay_ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (cfg.max_lr - min_lr)


def build_dataloaders(cfg: TrainingConfig, ddp_rank: int, ddp_world_size: int,
                      master_process: bool) -> Tuple[DataLoaderLite, DataLoaderLite]:
    common = dict(
        batch_size=cfg.micro_batch_size,
        seq_len=cfg.seq_len,
        process_rank=ddp_rank,
        world_size=ddp_world_size,
        data_root=cfg.data_root,
        master_process=master_process,
    )
    train_loader = DataLoaderLite(LoaderConfig(split="train", **common))
    val_loader = DataLoaderLite(LoaderConfig(split="val", **common))
    return train_loader, val_loader


def maybe_prepare_hellaswag(master_process: bool) -> bool:
    if iterate_examples is None or render_example is None:
        if master_process:
            print("hellaswag package not available; skipping hella evaluation")
        return False
    return True


def train(model_cfg: Optional[GPTConfig] = None, train_cfg: Optional[TrainingConfig] = None) -> None:
    model_cfg = model_cfg or GPTConfig(vocab_size=50304)
    train_cfg = train_cfg or TrainingConfig()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, master_process = setup_distributed()

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")

    grad_accum_steps = train_cfg.total_batch_size // (train_cfg.micro_batch_size * train_cfg.seq_len * ddp_world_size)
    if grad_accum_steps == 0 or train_cfg.total_batch_size % (train_cfg.micro_batch_size * train_cfg.seq_len * ddp_world_size) != 0:
        raise ValueError("total_batch_size must be divisible by micro_batch_size * seq_len * world_size")
    if master_process:
        print(f"total desired batch size: {train_cfg.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader, val_loader = build_dataloaders(train_cfg, ddp_rank, ddp_world_size, master_process)

    torch.set_float32_matmul_precision('high')

    model = GPT(model_cfg)
    model.to(device)
    if train_cfg.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(
        weight_decay=train_cfg.weight_decay,
        learning_rate=train_cfg.max_lr,
        device_type=device_type,
        master_process=master_process,
    )

    os.makedirs(train_cfg.log_dir, exist_ok=True)
    log_file = os.path.join(train_cfg.log_dir, "log.txt")
    with open(log_file, "w"):
        pass

    hellaswag_enabled = maybe_prepare_hellaswag(master_process)

    for step in range(train_cfg.max_steps):
        t0 = time.time()
        last_step = step == train_cfg.max_steps - 1

        if step % train_cfg.eval_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(train_cfg.eval_iters):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    loss = loss / train_cfg.eval_iters
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    checkpoint_path = os.path.join(train_cfg.log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                    }
                    torch.save(checkpoint, checkpoint_path)

        if (step % train_cfg.hellaswag_interval == 0 or last_step) and hellaswag_enabled and not train_cfg.compile:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        if ((step > 0 and step % train_cfg.sampling_interval == 0) or last_step) and not train_cfg.compile:
            model.eval()
            tokens = enc.encode(train_cfg.sample_prompt)
            tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            tokens = tokens.repeat(train_cfg.sample_count, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < train_cfg.sample_max_len:
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = model(xgen)
                    logits = logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
            for i in range(train_cfg.sample_count):
                tokens = xgen[i, :train_cfg.sample_max_len].tolist()
                decoded = enc.decode(tokens)
                if master_process:
                    print(f"rank {ddp_rank} sample {i}: {decoded}")

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    teardown_distributed(ddp)


if __name__ == "__main__":
    train()
