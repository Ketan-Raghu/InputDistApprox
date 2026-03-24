import json
import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self, log_dir: str, log_file: str = "training_log.jsonl", window_size: int = 50):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / log_file
        self.window_size = window_size

        self.loss_window = deque(maxlen=window_size)
        self.step_times = deque(maxlen=window_size)
        self.token_counts = deque(maxlen=window_size)

        self._last_step_time = None
        self._step_count = 0

    def start_step(self):
        self._last_step_time = time.time()

    def end_step(self, loss: float, num_tokens: int):
        elapsed = time.time() - self._last_step_time
        self.loss_window.append(loss)
        self.step_times.append(elapsed)
        self.token_counts.append(num_tokens)
        self._step_count += 1

    @property
    def avg_loss(self) -> float:
        if not self.loss_window:
            return 0.0
        valid = [x for x in self.loss_window if not math.isnan(x)]
        if not valid:
            return float("nan")
        return sum(valid) / len(valid)

    @property
    def tokens_per_sec(self) -> float:
        if not self.step_times:
            return 0.0
        total_tokens = sum(self.token_counts)
        total_time = sum(self.step_times)
        return total_tokens / total_time if total_time > 0 else 0.0

    def log_step(
        self,
        global_step: int,
        epoch: int,
        loss: float,
        lr: float,
        stage: str,
        grad_norm: Optional[float] = None,
    ):
        """Write a JSON line to the log file."""
        record = {
            "step": global_step,
            "epoch": epoch,
            "stage": stage,
            "loss": round(loss, 6),
            "avg_loss": round(self.avg_loss, 6),
            "lr": lr,
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
        if grad_norm is not None:
            record["grad_norm"] = round(grad_norm, 4)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_eval(self, global_step: int, val_loss: float, stage: str):
        """Log validation metrics."""
        record = {
            "step": global_step,
            "stage": stage,
            "val_loss": round(val_loss, 6),
            "val_ppl": round(math.exp(min(val_loss, 20)), 2),
            "type": "eval",
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(
            f"[Eval] step={global_step} val_loss={val_loss:.4f} "
            f"val_ppl={math.exp(min(val_loss, 20)):.2f}"
        )


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    global_step: int,
    epoch: int,
    config,
    checkpoint_name: str,
    accelerator=None,
):
    """Save training checkpoint."""
    save_dir = Path(config.output_dir) / checkpoint_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if accelerator is not None:
        from accelerate import DistributedType

        accelerator.wait_for_everyone()

        if accelerator.distributed_type == DistributedType.FSDP:
            # FSDP needs special state dict gathering
            accelerator.save_state(str(save_dir))
        elif accelerator.is_main_process:
            # DDP — unwrap and save plain state dict (loadable by Stage 2)
            unwrapped = accelerator.unwrap_model(model)
            torch.save(
                {
                    "model_state_dict": unwrapped.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                },
                save_dir / "checkpoint.pt",
            )
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
            },
            save_dir / "checkpoint.pt",
        )

    # Save metadata (main process only when distributed)
    if accelerator is None or accelerator.is_main_process:
        meta = {"global_step": global_step, "epoch": epoch, "checkpoint_name": checkpoint_name}
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Checkpoint saved: {save_dir}")


def load_checkpoint(checkpoint_path: str, model, optimizer=None, scheduler=None):
    """Load a Stage 1 checkpoint (non-FSDP)."""
    ckpt = torch.load(
        Path(checkpoint_path) / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    logger.info(
        f"Loaded checkpoint from {checkpoint_path} "
        f"(step={ckpt['global_step']}, epoch={ckpt['epoch']})"
    )
    return ckpt["global_step"], ckpt["epoch"]


@torch.no_grad()
def evaluate(model, dataloader, device=None, accelerator=None) -> float:
    """Run evaluation and return average loss."""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    for batch in dataloader:
        if accelerator is not None:
            # accelerator.prepare already moved data
            pass
        elif device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        if accelerator is not None:
            loss = accelerator.gather(loss).mean()

        loss_val = loss.item()
        if not math.isnan(loss_val):
            total_loss += loss_val
            total_steps += 1

    model.train()
    return total_loss / max(total_steps, 1)
