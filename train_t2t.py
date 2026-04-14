"""
T2T (Token-to-Token) Inverse Prediction Training Pipeline.

Train a partially-reinitialized Qwen2.5-7B-Instruct to predict the original input
prompt given a single model output. Uses DDP across 5 GPUs.

Training stages:
  Epoch 0 (stabilization): Only reinit layers 24-27 trainable, lr=2e-4
  Epochs 1-3 (full finetune): All params trainable, 2-tier LR (1e-4 / 1e-5)

Usage:
  torchrun --nproc_per_node=5 train_t2t.py [args]
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from reinit_qwen_layers import reinit_module

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/ketan/LLMs/models/Qwen_Qwen2.5-7B-Instruct/"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "t2t_training_pairs.jsonl")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "t2t")

QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
NUM_LAYERS = 28
NUM_REINIT_LAYERS = 4
INITIALIZER_RANGE = 0.02


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    model_path: str = MODEL_PATH
    data_path: str = DATA_PATH
    checkpoint_dir: str = CHECKPOINT_DIR
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 4
    stabilization_epochs: int = 1
    stabilization_lr: float = 2e-4
    lr_high: float = 1e-4
    lr_low: float = 1e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_len: int = 2048
    max_output_tokens: int = 512
    max_prompt_tokens: int = 1024
    val_split: float = 0.02
    log_every: int = 10
    eval_every_steps: int = 500
    save_every_steps: int = 1000
    seed: int = 42
    gradient_checkpointing: bool = True
    num_workers: int = 4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class T2TDataset(Dataset):
    """
    Each record: {"prompt": str, "output": str, ...}
    Formatted as chat:
      system: default Qwen prompt
      user: {model_output}
      assistant: {original_prompt}
    Loss only on assistant tokens (original prompt + final <|im_end|>).
    """

    def __init__(self, records: list[dict], tokenizer, config: TrainingConfig):
        self.records = records
        self.tokenizer = tokenizer
        self.config = config

        # Pre-tokenize the context template pieces for label masking
        # We need to know where the assistant content starts
        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        model_output = record["output"]
        original_prompt = record["prompt"]

        # Build the full conversation as the model will see it
        messages = [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": model_output},
            {"role": "assistant", "content": original_prompt},
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        # Build context (everything up to assistant content) for label masking
        context_messages = [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": model_output},
        ]
        context_text = self.tokenizer.apply_chat_template(
            context_messages, tokenize=False, add_generation_prompt=True,
        )

        # Tokenize both
        full_ids = self.tokenizer.encode(
            full_text, add_special_tokens=False,
            truncation=True, max_length=self.config.max_seq_len,
        )
        context_ids = self.tokenizer.encode(
            context_text, add_special_tokens=False,
            truncation=True, max_length=self.config.max_output_tokens + 128,  # output + template overhead
        )

        context_len = len(context_ids)

        # Labels: -100 for context, actual ids for assistant response
        labels = [-100] * context_len + full_ids[context_len:]

        # Ensure same length
        input_ids = full_ids
        if len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        elif len(labels) < len(input_ids):
            labels = labels + [-100] * (len(input_ids) - len(labels))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "context_len": context_len,
            "total_len": len(input_ids),
        }


def collate_fn(batch, pad_token_id: int):
    """Right-pad batch to max length."""
    max_len = max(item["total_len"] for item in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        seq_len = item["total_len"]
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def load_and_split_data(config: TrainingConfig, rank: int):
    """Load JSONL records and split into train/val."""
    records = []
    with open(config.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if rank == 0:
        print(f"Loaded {len(records)} records from {config.data_path}")

    # Deterministic shuffle + split
    rng = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(records), generator=rng).tolist()
    val_size = int(len(records) * config.val_split)
    val_indices = set(indices[:val_size])

    train_records = [records[i] for i in indices if i not in val_indices]
    val_records = [records[i] for i in indices if i in val_indices]

    if rank == 0:
        print(f"  {len(train_records)} train / {len(val_records)} val records")

    return train_records, val_records


def report_truncation_stats(dataset: T2TDataset, rank: int):
    """Print truncation statistics (rank 0 only)."""
    if rank != 0:
        return

    context_lens = []
    total_lens = []
    output_truncated = 0
    prompt_truncated = 0

    # Sample up to 10000 records for stats
    sample_size = min(len(dataset), 10000)
    for i in range(sample_size):
        item = dataset[i]
        ctx_len = item["context_len"]
        tot_len = item["total_len"]
        context_lens.append(ctx_len)
        total_lens.append(tot_len)

        # Check if output was likely truncated (context near max)
        if ctx_len >= dataset.config.max_output_tokens + 100:
            output_truncated += 1
        # Check if total hit max_seq_len
        if tot_len >= dataset.config.max_seq_len:
            prompt_truncated += 1

    context_lens_t = torch.tensor(context_lens, dtype=torch.float)
    total_lens_t = torch.tensor(total_lens, dtype=torch.float)

    print(f"\n{'=' * 50}")
    print(f"Truncation Report (sampled {sample_size}/{len(dataset)} records)")
    print(f"{'=' * 50}")
    print(f"  Context (system+user) tokens: avg={context_lens_t.mean():.0f}, "
          f"p95={context_lens_t.quantile(0.95):.0f}, max={context_lens_t.max():.0f}")
    print(f"  Total sequence tokens: avg={total_lens_t.mean():.0f}, "
          f"p95={total_lens_t.quantile(0.95):.0f}, max={total_lens_t.max():.0f}")
    print(f"  Output possibly truncated: {output_truncated}/{sample_size} "
          f"({100 * output_truncated / sample_size:.1f}%)")
    print(f"  Prompt possibly truncated: {prompt_truncated}/{sample_size} "
          f"({100 * prompt_truncated / sample_size:.1f}%)")

    if output_truncated / sample_size > 0.05:
        print(f"  WARNING: >5% outputs may be truncated. Consider increasing --max-output-tokens")
    if prompt_truncated / sample_size > 0.02:
        print(f"  WARNING: >2% prompts may be truncated. Consider increasing --max-seq-len")
    print(f"{'=' * 50}\n")


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def setup_model(config: TrainingConfig, local_rank: int):
    """Load model, reinitialize layers, wrap in DDP."""
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    if local_rank == 0:
        print(f"Loading model from {config.model_path} (attn={attn_impl})...")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    # Reinitialize last N layers
    start_layer = NUM_LAYERS - NUM_REINIT_LAYERS
    if local_rank == 0:
        print(f"  Reinitializing layers {start_layer}-{NUM_LAYERS - 1}")
    for idx in range(start_layer, NUM_LAYERS):
        reinit_module(model.model.layers[idx], INITIALIZER_RANGE)

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank])

    return model


def configure_stabilization(model: DDP):
    """Freeze all except reinit layers 24-27."""
    for p in model.parameters():
        p.requires_grad = False

    start_layer = NUM_LAYERS - NUM_REINIT_LAYERS
    for idx in range(start_layer, NUM_LAYERS):
        for p in model.module.model.layers[idx].parameters():
            p.requires_grad = True


def configure_full_finetune(model: DDP):
    """Unfreeze all parameters."""
    for p in model.parameters():
        p.requires_grad = True


def get_param_groups(model: DDP, config: TrainingConfig, stage: str) -> list[dict]:
    """Build optimizer parameter groups."""
    start_layer = NUM_LAYERS - NUM_REINIT_LAYERS

    if stage == "stabilization":
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params, "lr": config.stabilization_lr}]

    # Full finetune: 2-tier LR
    reinit_params = []
    pretrained_params = []
    for name, p in model.module.named_parameters():
        if not p.requires_grad:
            continue
        is_reinit = any(
            f"model.layers.{i}." in name for i in range(start_layer, NUM_LAYERS)
        )
        if is_reinit:
            reinit_params.append(p)
        else:
            pretrained_params.append(p)

    return [
        {"params": pretrained_params, "lr": config.lr_low, "name": "pretrained"},
        {"params": reinit_params, "lr": config.lr_high, "name": "reinitialized"},
    ]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model, dataloader, optimizer, scheduler,
    *, stage, epoch, global_step, config, log_path, local_rank, device,
):
    """Train for one epoch. Returns updated global_step."""
    model.train()
    running_loss = 0.0
    num_losses = 0
    optimizer.zero_grad()

    pbar = tqdm(
        enumerate(dataloader), total=len(dataloader),
        desc=f"{stage} Epoch {epoch}",
        disable=(local_rank != 0),
    )
    t_log = time.perf_counter()

    for batch_idx, batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = outputs.loss / config.gradient_accumulation_steps

        if torch.isnan(loss):
            if local_rank == 0:
                print(f"  WARNING: NaN loss at step {global_step}, skipping batch")
            optimizer.zero_grad()
            continue

        loss.backward()
        running_loss += loss.item() * config.gradient_accumulation_steps
        num_losses += 1

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            grad_norm = nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.max_grad_norm,
            ).item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging (rank 0 only)
            if local_rank == 0 and global_step % config.log_every == 0:
                now = time.perf_counter()
                elapsed = now - t_log
                t_log = now

                avg_loss = running_loss / max(num_losses, 1)
                lr = optimizer.param_groups[0]["lr"]
                gpu_mem = torch.cuda.memory_allocated(local_rank) / (1024 ** 3)

                entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "stage": stage,
                    "loss": round(loss.item() * config.gradient_accumulation_steps, 6),
                    "avg_loss": round(avg_loss, 6),
                    "lr": lr,
                    "grad_norm": round(grad_norm, 4),
                    "gpu_mem_gb": round(gpu_mem, 2),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    gn=f"{grad_norm:.2f}",
                    lr=f"{lr:.2e}",
                    mem=f"{gpu_mem:.1f}G",
                )

            # Periodic checkpoint (rank 0)
            if local_rank == 0 and global_step % config.save_every_steps == 0:
                save_checkpoint(
                    model, optimizer, scheduler,
                    stage, global_step, epoch, config,
                    os.path.join(config.checkpoint_dir, f"epoch{epoch}_step{global_step}.pt"),
                )

            # Periodic eval
            if global_step % config.eval_every_steps == 0:
                val_loss = validate(model, model._val_loader, device, local_rank)
                if local_rank == 0:
                    print(f"  Step {global_step} val_loss: {val_loss:.4f}")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "step": global_step,
                            "epoch": epoch,
                            "stage": f"{stage}_val",
                            "val_loss": round(val_loss, 6),
                        }) + "\n")

    return global_step


@torch.no_grad()
def validate(model, dataloader, device, local_rank, max_batches=50):
    """Run validation across DDP ranks. Returns average loss."""
    total_loss = 0.0
    count = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        if not torch.isnan(outputs.loss):
            total_loss += outputs.loss.item()
            count += 1

    # Reduce across ranks
    loss_tensor = torch.tensor([total_loss, count], dtype=torch.float64, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    total_loss_all, count_all = loss_tensor[0].item(), loss_tensor[1].item()

    return total_loss_all / max(count_all, 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler, stage, step, epoch, config, path):
    """Save checkpoint (rank 0 only — caller must guard)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "stage": stage,
            "step": step,
            "epoch": epoch,
            "config": asdict(config),
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, local_rank=0):
    """Load a checkpoint. Returns dict with stage/step/epoch."""
    if local_rank == 0:
        print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=f"cuda:{local_rank}", weights_only=False)

    model.module.load_state_dict(ckpt["model_state_dict"])
    if local_rank == 0:
        print(f"  Restored model weights (stage={ckpt['stage']}, step={ckpt['step']})")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if local_rank == 0:
            print(f"  Restored optimizer state")

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if local_rank == 0:
            print(f"  Restored scheduler state")

    return {
        "stage": ckpt["stage"],
        "step": ckpt["step"],
        "epoch": ckpt["epoch"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="T2T Inverse Prediction Training (DDP)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--stabilization-epochs", type=int, default=1)
    parser.add_argument("--stabilization-lr", type=float, default=2e-4)
    parser.add_argument("--lr-high", type=float, default=1e-4)
    parser.add_argument("--lr-low", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--val-split", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # DDP init
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    config = TrainingConfig(
        model_path=args.model_path,
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        stabilization_epochs=args.stabilization_epochs,
        stabilization_lr=args.stabilization_lr,
        lr_high=args.lr_high,
        lr_low=args.lr_low,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_seq_len=args.max_seq_len,
        max_output_tokens=args.max_output_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        val_split=args.val_split,
        log_every=args.log_every,
        eval_every_steps=args.eval_every_steps,
        save_every_steps=args.save_every_steps,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        num_workers=args.num_workers,
    )

    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if local_rank == 0:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    dist.barrier()

    log_path = os.path.join(config.checkpoint_dir, "training_log.jsonl")

    # -- Load tokenizer -----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # -- Load data ----------------------------------------------------------
    train_records, val_records = load_and_split_data(config, local_rank)
    train_dataset = T2TDataset(train_records, tokenizer, config)
    val_dataset = T2TDataset(val_records, tokenizer, config)

    report_truncation_stats(train_dataset, local_rank)

    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=config.seed)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    def collate_wrapper(batch):
        return collate_fn(batch, pad_token_id)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=train_sampler,
        collate_fn=collate_wrapper, num_workers=config.num_workers,
        pin_memory=True, persistent_workers=(config.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, sampler=val_sampler,
        collate_fn=collate_wrapper, num_workers=config.num_workers,
        pin_memory=True, persistent_workers=(config.num_workers > 0),
    )

    # -- Setup model --------------------------------------------------------
    model = setup_model(config, local_rank)
    model._val_loader = val_loader

    world_size = dist.get_world_size()
    effective_batch = config.batch_size * config.gradient_accumulation_steps * world_size
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_params = sum(p.numel() for p in model.parameters())

    if local_rank == 0:
        print(f"\n{'=' * 60}")
        print(f"T2T Training Configuration")
        print(f"{'=' * 60}")
        print(f"  World size: {world_size} GPUs")
        print(f"  Effective batch: {config.batch_size} x {config.gradient_accumulation_steps} x {world_size} = {effective_batch}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total params: {total_params:,}")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"{'=' * 60}\n")

    # -- Resume handling ----------------------------------------------------
    resume_epoch = 0
    resume_step = 0
    resume_stage = None
    if args.resume_from:
        ckpt_info = load_checkpoint(args.resume_from, model, local_rank=local_rank)
        resume_stage = ckpt_info["stage"]
        resume_step = ckpt_info["step"]
        resume_epoch = ckpt_info["epoch"]
        if local_rank == 0:
            print(f"  Resumed: stage={resume_stage}, epoch={resume_epoch}, step={resume_step}")

    # ======================================================================
    # Stage 1: Stabilization — only reinit layers 24-27
    # ======================================================================
    global_step = 0

    if resume_stage != "full_finetune":
        if local_rank == 0:
            print(f"\n{'=' * 60}")
            print(f"STABILIZATION: Epochs 0..{config.stabilization_epochs - 1} (reinit layers only)")
            print(f"{'=' * 60}")

        configure_stabilization(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if local_rank == 0:
            print(f"  Trainable: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.1f}%)")

        param_groups = get_param_groups(model, config, stage="stabilization")
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

        total_stab_steps = steps_per_epoch * config.stabilization_epochs
        warmup_steps = int(total_stab_steps * config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_stab_steps)

        for epoch in range(config.stabilization_epochs):
            train_sampler.set_epoch(epoch)
            global_step = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                stage="stabilization", epoch=epoch, global_step=global_step,
                config=config, log_path=log_path, local_rank=local_rank, device=device,
            )

            # End-of-epoch checkpoint
            if local_rank == 0:
                save_checkpoint(
                    model, optimizer, scheduler,
                    "stabilization", global_step, epoch, config,
                    os.path.join(config.checkpoint_dir, f"epoch{epoch}_final.pt"),
                )
            dist.barrier()

    # ======================================================================
    # Stage 2: Full fine-tuning — all params with 2-tier LR
    # ======================================================================
    if local_rank == 0:
        print(f"\n{'=' * 60}")
        print(f"FULL FINETUNE: Epochs {config.stabilization_epochs}..{config.num_epochs - 1}")
        print(f"  Reinit layers lr={config.lr_high}, pretrained lr={config.lr_low}")
        print(f"{'=' * 60}")

    configure_full_finetune(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if local_rank == 0:
        print(f"  Trainable: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.1f}%)")

    param_groups = get_param_groups(model, config, stage="full_finetune")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    ft_epochs = config.num_epochs - config.stabilization_epochs
    total_ft_steps = steps_per_epoch * ft_epochs
    warmup_steps = int(total_ft_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_ft_steps)

    for epoch in range(config.stabilization_epochs, config.num_epochs):
        train_sampler.set_epoch(epoch)
        global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            stage="full_finetune", epoch=epoch, global_step=global_step,
            config=config, log_path=log_path, local_rank=local_rank, device=device,
        )

        # End-of-epoch checkpoint
        if local_rank == 0:
            save_checkpoint(
                model, optimizer, scheduler,
                "full_finetune", global_step, epoch, config,
                os.path.join(config.checkpoint_dir, f"epoch{epoch}_final.pt"),
            )
        dist.barrier()

    # -- Save final model (HuggingFace format) ------------------------------
    if local_rank == 0:
        final_dir = os.path.join(config.checkpoint_dir, "final_model")
        print(f"\nSaving final model to {final_dir}...")
        model.module.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print("Done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
