"""
Training From Scratch — Shared Training Engine
Handles training loop, DDP multi-GPU, mixed precision, gradient accumulation,
checkpointing, and logging for all three models.

Multi-GPU: Uses PyTorch DistributedDataParallel (DDP).
Launch with:  torchrun --nproc_per_node=N script.py
Single GPU:   python script.py  (works without any changes)
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path

from config_scratch import CHECKPOINT_DIR, LOG_DIR, SEED, DTYPE, COMPILE_MODEL

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ═══════════════════════════════════════════════════════════
#  DISTRIBUTED HELPERS
# ═══════════════════════════════════════════════════════════

def setup_distributed():
    """Initialize DDP if launched with torchrun. Returns (rank, world_size, device)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            print(f"  DDP initialized: {world_size} GPUs")
            print(f"  Device: {torch.cuda.get_device_name(local_rank)}")
        return rank, world_size, device
    else:
        # Single GPU or CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"  Single GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  VRAM: {vram:.1f} GB")
        else:
            device = torch.device("cpu")
            print("  Using CPU (training will be very slow)")
        return 0, 1, device


def cleanup_distributed():
    """Clean up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    return rank == 0


# ═══════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m"


def get_amp_dtype(device: torch.device):
    """Auto-detect best mixed precision dtype for the GPU."""
    if DTYPE != "auto":
        return torch.bfloat16 if DTYPE == "bfloat16" else torch.float16

    if device.type != "cuda":
        return torch.float32

    # A100/H100/H200 support bf16 natively
    capability = torch.cuda.get_device_capability(device)
    if capability[0] >= 8:  # Ampere+
        return torch.bfloat16
    return torch.float16


def get_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01):
    """AdamW with weight decay only on non-bias/norm params."""
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm"}
    params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def get_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Linear warmup + cosine decay."""
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(num_training_steps - num_warmup_steps, 1))
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[num_warmup_steps])


# ═══════════════════════════════════════════════════════════
#  TRAINER
# ═══════════════════════════════════════════════════════════

class Trainer:
    """
    General-purpose trainer for from-scratch transformer models.
    Supports single-GPU and multi-GPU (DDP) training transparently.

    Usage:
        rank, world_size, device = setup_distributed()
        trainer = Trainer(model, "my_model", config, device, rank, world_size)
        trainer.train_loop(train_loader, val_loader, ...)
        cleanup_distributed()
    """

    def __init__(self, model: nn.Module, model_name: str, config: dict,
                 device: torch.device = None, rank: int = 0, world_size: int = 1):
        self.model_name = model_name
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        model = model.to(self.device)

        # Wrap with DDP if multi-GPU
        if world_size > 1:
            self.model = DDP(model, device_ids=[self.device.index],
                             output_device=self.device.index,
                             find_unused_parameters=False)
            self._raw_model = model  # keep reference for saving
        else:
            self.model = model
            self._raw_model = model

        # Mixed precision
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = get_amp_dtype(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        # torch.compile (optional)
        if COMPILE_MODEL and hasattr(torch, "compile") and world_size <= 1:
            try:
                self.model = torch.compile(self.model)
                if is_main_process(rank):
                    print("  Model compiled with torch.compile()")
            except Exception:
                if is_main_process(rank):
                    print("  torch.compile() not available, using eager mode")

        # Logging
        self.log_file = LOG_DIR / f"{model_name}_training.jsonl"
        self.global_step = 0
        self.best_val_loss = float("inf")

        if is_main_process(rank):
            param_count = count_parameters(self._raw_model)
            print(f"  Model: {model_name}")
            print(f"  Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")
            print(f"  AMP dtype: {self.amp_dtype}")
            if world_size > 1:
                print(f"  DDP: {world_size} GPUs")

    def _log(self, metrics: dict):
        if not is_main_process(self.rank):
            return
        metrics["step"] = self.global_step
        metrics["timestamp"] = time.time()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    def save_checkpoint(self, tag: str = "latest"):
        """Save checkpoint (only on rank 0)."""
        if not is_main_process(self.rank):
            return
        save_dir = CHECKPOINT_DIR / self.model_name / tag
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self._raw_model.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, save_dir / "checkpoint.pt")
        print(f"  Checkpoint saved: {save_dir}")

    def load_checkpoint(self, tag: str = "latest") -> bool:
        """Load checkpoint if it exists."""
        ckpt_path = CHECKPOINT_DIR / self.model_name / tag / "checkpoint.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self._raw_model.load_state_dict(ckpt["model_state_dict"])
            self.global_step = ckpt.get("global_step", 0)
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            if is_main_process(self.rank):
                print(f"  Resumed from checkpoint: step {self.global_step}")
            return True
        return False

    def train_epoch(self, dataloader, optimizer, scheduler, grad_accum_steps: int = 1,
                    max_grad_norm: float = 1.0, forward_fn=None, epoch: int = 0,
                    max_steps: int = None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        log_interval = 100

        # Set epoch for DistributedSampler
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                if forward_fn:
                    loss = forward_fn(self.model, batch)
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                loss = loss / grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * grad_accum_steps
            num_batches += 1

            if is_main_process(self.rank) and num_batches % log_interval == 0:
                avg_loss = total_loss / num_batches
                elapsed = time.time() - start_time
                samples_per_sec = (num_batches * dataloader.batch_size) / elapsed
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"    Epoch {epoch} | Step {self.global_step} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                    f"Speed: {samples_per_sec:.0f} samples/s | "
                    f"Elapsed: {format_time(elapsed)}"
                )
                self._log({"phase": "train", "epoch": epoch, "loss": avg_loss,
                           "lr": lr, "samples_per_sec": samples_per_sec})

            if max_steps and self.global_step >= max_steps:
                if is_main_process(self.rank):
                    print(f"    Reached max_steps={max_steps}, stopping epoch early.")
                break

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader, forward_fn=None):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                if forward_fn:
                    loss = forward_fn(self.model, batch)
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self._log({"phase": "eval", "loss": avg_loss})
        return avg_loss

    def train_loop(self, train_loader, val_loader, epochs: int, lr: float,
                   grad_accum_steps: int = 1, max_grad_norm: float = 1.0,
                   warmup_ratio: float = 0.1, forward_fn=None,
                   max_steps: int = None, save_every_epoch: bool = True):
        """Full training loop with validation and checkpointing."""
        num_training_steps = len(train_loader) * epochs // grad_accum_steps
        if max_steps:
            num_training_steps = min(num_training_steps, max_steps)
        num_warmup_steps = int(num_training_steps * warmup_ratio)

        optimizer = get_optimizer(self._raw_model, lr)
        scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)

        if is_main_process(self.rank):
            eff_batch = train_loader.batch_size * grad_accum_steps * self.world_size
            print(f"\n  Training: {epochs} epochs, {num_training_steps} steps, "
                  f"{num_warmup_steps} warmup")
            print(f"  Gradient accumulation: {grad_accum_steps}")
            print(f"  Effective batch size: {eff_batch} "
                  f"({train_loader.batch_size} × {grad_accum_steps} × {self.world_size} GPUs)")

        for epoch in range(epochs):
            if is_main_process(self.rank):
                print(f"\n  ── Epoch {epoch + 1}/{epochs} ──")

            train_loss = self.train_epoch(
                train_loader, optimizer, scheduler, grad_accum_steps,
                max_grad_norm, forward_fn, epoch + 1, max_steps
            )

            val_loss = self.evaluate(val_loader, forward_fn)

            if is_main_process(self.rank):
                print(f"    Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")
                    print(f"    New best model! Val loss: {val_loss:.4f}")
                if save_every_epoch:
                    self.save_checkpoint(f"epoch_{epoch + 1}")

            # Sync barrier for DDP
            if self.world_size > 1:
                dist.barrier()

            if max_steps and self.global_step >= max_steps:
                if is_main_process(self.rank):
                    print(f"  Reached max_steps={max_steps}, stopping training.")
                break

        self.save_checkpoint("final")
        if is_main_process(self.rank):
            print(f"\n  Training complete. Best val loss: {self.best_val_loss:.4f}")
