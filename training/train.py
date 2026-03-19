"""
Inverse mapping training: teach LLaMA-3.1-8B to predict human prompts from model responses.

Usage:
    # Stage 1: Warmup reinitialized layers (DDP across 5 GPUs)
    accelerate launch --config_file accelerate_config_stage1.yaml training/train.py --stage warmup

    # Stage 2: Full FSDP training on 5 GPUs
    accelerate launch --config_file accelerate_config.yaml training/train.py --stage full --resume_from checkpoints/inverse_mapping/stage1_final
"""

import argparse
import logging
import math
import os
import sys

import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from training.config import TrainingConfig
from training.data import build_datasets, build_dataloader
from training.model_setup import (
    load_and_prepare_model,
    reinitialize_layers,
    freeze_for_warmup,
    unfreeze_all,
    build_optimizer_groups,
)
from training.utils import (
    MetricsTracker,
    save_checkpoint,
    load_checkpoint,
    evaluate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def count_tokens_in_batch(batch: dict) -> int:
    """Count non-padding tokens in a batch."""
    return batch["attention_mask"].sum().item()


def train_stage1(config: TrainingConfig):
    """Stage 1: Warmup reinitialized layers with DDP across multiple GPUs."""
    from accelerate import Accelerator

    logger.info("=" * 60)
    logger.info("STAGE 1: WARMUP — Reinitialized layers only (DDP)")
    logger.info("=" * 60)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.stage1_grad_accum,
        mixed_precision="bf16",
    )

    if accelerator.is_main_process:
        logger.info(f"Num processes: {accelerator.num_processes}")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Load model
    model, tokenizer = load_and_prepare_model(config)
    reinitialize_layers(model, config.reinit_layers, std=config.initializer_range)
    freeze_for_warmup(model, config.reinit_layers)

    # Build data
    train_dataset, val_dataset = build_datasets(config, tokenizer)
    train_loader = build_dataloader(
        train_dataset, config,
        batch_size=config.stage1_per_gpu_batch,
        shuffle=True,
        distributed=True,
    )
    val_loader = build_dataloader(
        val_dataset, config,
        batch_size=config.stage1_per_gpu_batch,
        shuffle=False,
        distributed=True,
    )

    # Optimizer + scheduler
    optimizer = build_optimizer_groups(model, config, stage="warmup")

    num_update_steps_per_epoch = len(train_loader) // config.stage1_grad_accum
    num_training_steps = num_update_steps_per_epoch * config.stage1_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare with accelerator (handles DDP wrapping)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    if accelerator.is_main_process:
        logger.info(
            f"Training steps: {num_training_steps}, warmup: {num_warmup_steps}, "
            f"batches/epoch: {len(train_loader)}"
        )

    # Metrics (main process only)
    metrics = MetricsTracker(config.output_dir, config.log_file) if accelerator.is_main_process else None

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(config.stage1_epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Stage1 Epoch {epoch + 1}/{config.stage1_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch_idx, batch in pbar:
            if metrics:
                metrics.start_step()

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                step_loss = loss.item()

                if metrics:
                    num_tokens = count_tokens_in_batch(batch) * accelerator.num_processes
                    metrics.end_step(step_loss, num_tokens)

                    pbar.set_postfix(
                        loss=f"{metrics.avg_loss:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        tps=f"{metrics.tokens_per_sec:.0f}",
                    )

                    if global_step % config.log_every == 0:
                        gn = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
                        metrics.log_step(
                            global_step, epoch, step_loss,
                            scheduler.get_last_lr()[0], "warmup",
                            grad_norm=gn,
                        )

                if global_step % config.checkpoint_every == 0:
                    val_loss = evaluate(model, val_loader, accelerator=accelerator)
                    if metrics:
                        metrics.log_eval(global_step, val_loss, "warmup")
                    save_checkpoint(
                        model, optimizer, scheduler,
                        global_step, epoch, config,
                        f"stage1_step{global_step}",
                        accelerator=accelerator,
                    )

    # Final evaluation + save
    val_loss = evaluate(model, val_loader, accelerator=accelerator)
    if metrics:
        metrics.log_eval(global_step, val_loss, "warmup")
        logger.info(f"Stage 1 final val_loss={val_loss:.4f}, ppl={math.exp(min(val_loss, 20)):.2f}")

    save_checkpoint(
        model, optimizer, scheduler,
        global_step, epoch, config,
        "stage1_final",
        accelerator=accelerator,
    )
    if accelerator.is_main_process:
        logger.info("Stage 1 complete.")


def train_stage2(config: TrainingConfig, resume_from: str):
    """Stage 2: Full training with FSDP on multiple GPUs."""
    from accelerate import Accelerator

    logger.info("=" * 60)
    logger.info("STAGE 2: FULL TRAINING — All parameters, FSDP")
    logger.info("=" * 60)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.stage2_grad_accum,
        mixed_precision="bf16",
    )

    if accelerator.is_main_process:
        logger.info(f"Num processes: {accelerator.num_processes}")
        logger.info(f"Resuming from: {resume_from}")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Load model + restore Stage 1 weights
    # Only load checkpoint on rank 0 — fsdp_sync_module_states broadcasts
    # rank 0's weights to all ranks during prepare()
    model, tokenizer = load_and_prepare_model(config)

    if resume_from and accelerator.is_main_process:
        load_checkpoint(resume_from, model)

    unfreeze_all(model)

    # Build data
    train_dataset, val_dataset = build_datasets(config, tokenizer)
    train_loader = build_dataloader(
        train_dataset, config,
        batch_size=config.stage2_per_gpu_batch,
        shuffle=True,
        distributed=True,
    )
    val_loader = build_dataloader(
        val_dataset, config,
        batch_size=config.stage2_per_gpu_batch,
        shuffle=False,
        distributed=True,
    )

    # Prepare model first — FSDP shards parameters across ranks,
    # reducing per-rank memory before optimizer allocation
    model = accelerator.prepare(model)

    # Optimizer + scheduler (on FSDP-sharded parameters)
    optimizer = build_optimizer_groups(model, config, stage="full")

    num_update_steps_per_epoch = len(train_loader) // config.stage2_grad_accum
    num_training_steps = num_update_steps_per_epoch * config.stage2_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare remaining components
    optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        optimizer, train_loader, val_loader, scheduler
    )

    if accelerator.is_main_process:
        logger.info(
            f"Training steps: {num_training_steps}, warmup: {num_warmup_steps}, "
            f"batches/epoch: {len(train_loader)}"
        )

    # Metrics (main process only)
    metrics = MetricsTracker(config.output_dir, config.log_file) if accelerator.is_main_process else None

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(config.stage2_epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Stage2 Epoch {epoch + 1}/{config.stage2_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch_idx, batch in pbar:
            if metrics:
                metrics.start_step()

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                step_loss = loss.item()

                if metrics:
                    num_tokens = count_tokens_in_batch(batch) * accelerator.num_processes
                    metrics.end_step(step_loss, num_tokens)

                    pbar.set_postfix(
                        loss=f"{metrics.avg_loss:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        tps=f"{metrics.tokens_per_sec:.0f}",
                    )

                    if global_step % config.log_every == 0:
                        gn = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
                        metrics.log_step(
                            global_step, epoch, step_loss,
                            scheduler.get_last_lr()[0], "full",
                            grad_norm=gn,
                        )

                if global_step % config.checkpoint_every == 0:
                    val_loss = evaluate(model, val_loader, accelerator=accelerator)
                    if metrics:
                        metrics.log_eval(global_step, val_loss, "full")
                    save_checkpoint(
                        model, optimizer, scheduler,
                        global_step, epoch, config,
                        f"stage2_step{global_step}",
                        accelerator=accelerator,
                    )

        # End-of-epoch eval
        val_loss = evaluate(model, val_loader, accelerator=accelerator)
        if metrics:
            metrics.log_eval(global_step, val_loss, "full")
            logger.info(
                f"Epoch {epoch + 1} val_loss={val_loss:.4f}, "
                f"ppl={math.exp(min(val_loss, 20)):.2f}"
            )

    # Save final model in HuggingFace format
    if accelerator.is_main_process:
        logger.info("Saving final model in HuggingFace format...")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        os.path.join(config.output_dir, "final_model"),
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(os.path.join(config.output_dir, "final_model"))
        logger.info("Stage 2 complete. Final model saved.")


def main():
    parser = argparse.ArgumentParser(description="Inverse mapping training")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["warmup", "full"],
        help="Training stage: 'warmup' (Stage 1) or 'full' (Stage 2)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (required for Stage 2)",
    )
    args = parser.parse_args()

    config = TrainingConfig()

    if args.stage == "warmup":
        train_stage1(config)
    elif args.stage == "full":
        if args.resume_from is None:
            logger.warning(
                "No --resume_from specified for Stage 2. "
                "Starting from base model without Stage 1 warmup."
            )
        train_stage2(config, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
