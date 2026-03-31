"""
L2L (Learn-to-Learn) Training Pipeline.

Three Qwen2.5-7B-Instruct model roles form a differentiable chain:
  Model A (frozen) -> Middle Model (trainable) -> Model B (frozen)

Model A generates soft probability sequences from prompts. The Middle Model
(with reinitialized embed_tokens and last 8 layers) transforms them. Model B
reconstructs Model A's output. Loss is shifted soft cross-entropy.

Two-stage training:
  Stage 1: Train only reinitialized layers (embed_tokens + layers 20-27)
  Stage 2: Full finetune with 2-tier LR, temperature annealing toward discrete

Usage:
  python l2l_pipeline.py [--batch-size 2] [--stage1-epochs 1] [--stage2-epochs 3]
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from reinit_qwen_layers import reinit_module

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/ketan/LLMs/models/Qwen_Qwen2.5-7B-Instruct/"
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "data", "merged_prompts.jsonl")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "l2l")

IM_START_ID = 151644
IM_END_ID = 151645
ASSISTANT_ID = 77091
USER_ID = 872
VOCAB_SIZE = 152064
NUM_LAYERS = 28
NUM_REINIT_LAYERS = 8
INITIALIZER_RANGE = 0.02


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    model_path: str = MODEL_PATH
    data_path: str = PROMPTS_PATH
    checkpoint_dir: str = CHECKPOINT_DIR
    max_new_tokens: int = 256
    max_prompt_tokens: int = 1024
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    stage1_epochs: int = 1
    stage2_epochs: int = 3
    stage1_lr: float = 2e-4
    stage2_lr_high: float = 1e-4
    stage2_lr_low: float = 1e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    stage1_temperature: float = 1.0
    stage2_temp_start: float = 1.0
    stage2_temp_end: float = 0.05
    temp_schedule: str = "cosine"
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
class PromptDataset(Dataset):
    """Simple dataset that loads all prompts from the JSONL file into memory."""

    def __init__(self, path: str):
        self.prompts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.prompts.append(json.loads(line)["prompt"])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def make_collate_fn(tokenizer, max_prompt_tokens: int):
    """Return a collate function that tokenizes a batch of raw prompt strings."""

    def collate_fn(batch: list[str]):
        messages_list = [[{"role": "user", "content": p}] for p in batch]
        texts = [
            tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True,
            )
            for m in messages_list
        ]
        encodings = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        return encodings

    return collate_fn


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class L2LPipeline(nn.Module):
    """
    Full L2L pipeline: Model A -> Middle Model -> Model B.

    Model A and Model B share one frozen Qwen2.5-7B instance.
    The Middle Model is a separate instance with reinitialized embed_tokens
    and layers 20-27; its lm_head is always frozen.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.dtype = torch.bfloat16
        self._load_models()

    # -- model setup --------------------------------------------------------

    def _get_device(self, model):
        """Device of the model's embedding layer (entry point for inputs)."""
        return model.model.embed_tokens.weight.device

    def _load_models(self):
        cfg = self.config

        # Base model — shared for Model A (generation) and Model B (reconstruction)
        print("Loading base model (frozen, shared for Model A & B)...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, torch_dtype=self.dtype,
            device_map="auto", trust_remote_code=True,
        )
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.eval()

        # Tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Middle model — separate instance, partially reinitialized
        print("Loading middle model...")
        self.middle_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, torch_dtype=self.dtype,
            device_map="auto", trust_remote_code=True,
        )

        # Reinitialize embed_tokens
        print("Reinitializing middle model embed_tokens...")
        nn.init.normal_(
            self.middle_model.model.embed_tokens.weight,
            mean=0.0, std=INITIALIZER_RANGE,
        )

        # Reinitialize last 8 transformer layers
        start = NUM_LAYERS - NUM_REINIT_LAYERS
        print(f"Reinitializing middle model layers {start}-{NUM_LAYERS - 1}...")
        for idx in range(start, NUM_LAYERS):
            reinit_module(self.middle_model.model.layers[idx], INITIALIZER_RANGE)

        # lm_head is always frozen
        for p in self.middle_model.lm_head.parameters():
            p.requires_grad = False

        if cfg.gradient_checkpointing:
            self.middle_model.gradient_checkpointing_enable()

    # -- stage configuration ------------------------------------------------

    def configure_stage1(self):
        """Stage 1: only train reinitialized params (embed_tokens + layers 20-27)."""
        for p in self.middle_model.parameters():
            p.requires_grad = False
        for p in self.middle_model.model.embed_tokens.parameters():
            p.requires_grad = True
        start = NUM_LAYERS - NUM_REINIT_LAYERS
        for idx in range(start, NUM_LAYERS):
            for p in self.middle_model.model.layers[idx].parameters():
                p.requires_grad = True

    def configure_stage2(self):
        """Stage 2: unfreeze all middle model params except lm_head."""
        for p in self.middle_model.parameters():
            p.requires_grad = True
        for p in self.middle_model.lm_head.parameters():
            p.requires_grad = False

    def get_param_groups(self, stage: int) -> list[dict]:
        """Return optimizer parameter groups for the given stage."""
        cfg = self.config
        start = NUM_LAYERS - NUM_REINIT_LAYERS

        if stage == 1:
            params = [p for p in self.middle_model.parameters() if p.requires_grad]
            return [{"params": params, "lr": cfg.stage1_lr}]

        # Stage 2: 2-tier learning rate
        reinit_params, pretrained_params = [], []
        for name, p in self.middle_model.named_parameters():
            if not p.requires_grad:
                continue
            is_reinit = name.startswith("model.embed_tokens") or any(
                f"model.layers.{i}." in name for i in range(start, NUM_LAYERS)
            )
            (reinit_params if is_reinit else pretrained_params).append(p)

        return [
            {"params": pretrained_params, "lr": cfg.stage2_lr_low, "name": "pretrained"},
            {"params": reinit_params, "lr": cfg.stage2_lr_high, "name": "reinitialized"},
        ]

    # -- forward components -------------------------------------------------

    @torch.no_grad()
    def generate_model_a(self, input_ids, attention_mask):
        """
        Model A autoregressive generation.
        Returns raw_logits (batch, gen_steps, vocab) and gen_lengths (batch,).
        """
        device = self._get_device(self.base_model)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        gen_output = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            min_new_tokens=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            output_logits=True,
            return_dict_in_generate=True,
        )

        raw_logits = torch.stack(gen_output.logits, dim=1)  # (B, gen, V)

        # Find per-sequence generation length (first <|im_end|>, exclusive)
        prompt_len = input_ids.shape[1]
        gen_tokens = gen_output.sequences[:, prompt_len:]
        is_eos = gen_tokens == IM_END_ID
        has_eos = is_eos.any(dim=1)
        first_eos = is_eos.int().argmax(dim=1)
        gen_lengths = torch.where(
            has_eos, first_eos,
            torch.full_like(first_eos, gen_tokens.shape[1]),
        ).clamp(min=1)

        torch.cuda.empty_cache()
        return raw_logits, gen_lengths

    def build_soft_sequence(self, raw_logits, gen_lengths, temperature):
        """
        Build padded soft probability sequence with <|im_start|>assistant prefix
        and <|im_start|>user suffix.

        Returns:
            prob_vectors: (B, max_seq, V) — probability targets / middle input
            mask:         (B, max_seq)    — 1 for real positions, 0 for padding
        """
        B = raw_logits.shape[0]
        device = raw_logits.device
        temp = max(temperature, 1e-6)

        probs = F.softmax(raw_logits / temp, dim=-1)

        max_gen = gen_lengths.max().item()
        max_total = max_gen + 4  # 2 prefix + gen + 2 suffix

        output = torch.zeros(B, max_total, VOCAB_SIZE, device=device, dtype=self.dtype)
        mask = torch.zeros(B, max_total, device=device, dtype=self.dtype)

        for i in range(B):
            gl = gen_lengths[i].item()
            pos = 0

            # Prefix: <|im_start|> assistant
            output[i, pos, IM_START_ID] = 1.0
            pos += 1
            output[i, pos, ASSISTANT_ID] = 1.0
            pos += 1

            # Model A probability vectors (truncated before EOS)
            output[i, pos : pos + gl] = probs[i, :gl]
            pos += gl

            # Suffix: <|im_start|> user
            output[i, pos, IM_START_ID] = 1.0
            pos += 1
            output[i, pos, USER_ID] = 1.0
            pos += 1

            mask[i, :pos] = 1.0

        return output, mask

    def forward(self, input_ids, attention_mask, temperature):
        """
        Full forward: Model A generation -> Middle Model -> Model B -> loss.
        """
        # 1. Model A generation (detached, no grad)
        raw_logits, gen_lengths = self.generate_model_a(input_ids, attention_mask)

        # 2. Build soft probability sequence with special-token framing
        prob_vectors, mask = self.build_soft_sequence(
            raw_logits, gen_lengths, temperature,
        )

        # 3. Middle model: soft embed -> transformer layers -> lm_head -> softmax
        mid_device = self._get_device(self.middle_model)
        mid_embed = self.middle_model.model.embed_tokens.weight
        soft_embeds = prob_vectors.to(mid_device) @ mid_embed

        middle_out = self.middle_model(
            inputs_embeds=soft_embeds,
            attention_mask=mask.to(mid_device),
            use_cache=False,
        )
        middle_probs = F.softmax(middle_out.logits, dim=-1)

        # 4. Model B: weighted embed -> frozen transformer -> logits
        base_device = self._get_device(self.base_model)
        base_embed = self.base_model.model.embed_tokens.weight
        weighted_embeds = middle_probs.to(base_device) @ base_embed

        model_b_out = self.base_model(
            inputs_embeds=weighted_embeds,
            attention_mask=mask.to(base_device),
            use_cache=False,
        )

        # 5. Shifted autoregressive soft cross-entropy loss
        b_logits = model_b_out.logits[:, :-1, :]
        targets = prob_vectors[:, 1:, :].to(b_logits.device)
        shifted_mask = mask[:, 1:].to(b_logits.device)

        log_probs = F.log_softmax(b_logits, dim=-1)
        per_token_loss = -(targets * log_probs).sum(dim=-1)
        loss = (per_token_loss * shifted_mask).sum() / shifted_mask.sum().clamp(min=1)

        return loss


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def get_temperature(step, total_steps, schedule="cosine", start=1.0, end=0.05):
    """Compute softmax temperature at the given training step."""
    if total_steps <= 0:
        return start
    progress = min(step / total_steps, 1.0)
    if schedule == "cosine":
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
    return start + (end - start) * progress  # linear


def save_checkpoint(pipeline, optimizer, scheduler, stage, step, epoch, temp, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "middle_model_state_dict": pipeline.middle_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "stage": stage,
            "step": step,
            "epoch": epoch,
            "temperature": temp,
            "config": asdict(pipeline.config),
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")


def train_one_epoch(
    pipeline, dataloader, optimizer, scheduler,
    *, stage, epoch, global_step, temperature_fn, log_path, config,
):
    """Train for one epoch. Returns updated global_step."""
    pipeline.middle_model.train()
    running_loss = 0.0
    num_losses = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Stage {stage} Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        temp = temperature_fn(global_step) if callable(temperature_fn) else temperature_fn
        t0 = time.time()

        loss = pipeline(batch["input_ids"], batch["attention_mask"], temp)
        loss = loss / config.gradient_accumulation_steps

        if torch.isnan(loss):
            print(f"  WARNING: NaN loss at step {global_step}, skipping")
            optimizer.zero_grad()
            continue

        loss.backward()
        running_loss += loss.item() * config.gradient_accumulation_steps
        num_losses += 1

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            grad_norm = nn.utils.clip_grad_norm_(
                [p for p in pipeline.middle_model.parameters() if p.requires_grad],
                config.max_grad_norm,
            ).item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging
            if global_step % config.log_every == 0:
                avg_loss = running_loss / max(num_losses, 1)
                lr = optimizer.param_groups[0]["lr"]
                gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                elapsed = time.time() - t0

                entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "stage": f"stage{stage}",
                    "loss": round(loss.item() * config.gradient_accumulation_steps, 6),
                    "avg_loss": round(avg_loss, 6),
                    "lr": lr,
                    "grad_norm": round(grad_norm, 4),
                    "gpu_mem_gb": round(gpu_mem, 2),
                    "temperature": round(temp, 5),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                pbar.set_postfix(loss=f"{avg_loss:.4f}", temp=f"{temp:.3f}", gn=f"{grad_norm:.2f}")

            # Periodic checkpoint
            if global_step % config.save_every_steps == 0:
                ckpt = os.path.join(config.checkpoint_dir, f"stage{stage}_step{global_step}.pt")
                save_checkpoint(pipeline, optimizer, scheduler, stage, global_step, epoch, temp, ckpt)

            # Periodic eval
            if global_step % config.eval_every_steps == 0:
                val_loss = validate(pipeline, pipeline._val_loader, temp)
                print(f"  Step {global_step} val_loss: {val_loss:.4f}")
                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "step": global_step, "epoch": epoch,
                        "stage": f"stage{stage}_val", "val_loss": round(val_loss, 6),
                        "temperature": round(temp, 5),
                    }) + "\n")
                pipeline.middle_model.train()

    return global_step


@torch.no_grad()
def validate(pipeline, dataloader, temperature, max_batches=50):
    """Run validation, returns average loss."""
    pipeline.middle_model.eval()
    total_loss = 0.0
    count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
        if batch_idx >= max_batches:
            break
        loss = pipeline(batch["input_ids"], batch["attention_mask"], temperature)
        if not torch.isnan(loss):
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="L2L Training Pipeline")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--stage1-epochs", type=int, default=1)
    parser.add_argument("--stage2-epochs", type=int, default=3)
    parser.add_argument("--stage1-lr", type=float, default=2e-4)
    parser.add_argument("--stage2-lr-high", type=float, default=1e-4)
    parser.add_argument("--stage2-lr-low", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--temp-end", type=float, default=0.05)
    parser.add_argument("--temp-schedule", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    config = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_lr=args.stage1_lr,
        stage2_lr_high=args.stage2_lr_high,
        stage2_lr_low=args.stage2_lr_low,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        log_every=args.log_every,
        eval_every_steps=args.eval_every_steps,
        save_every_steps=args.save_every_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        stage2_temp_end=args.temp_end,
        temp_schedule=args.temp_schedule,
        num_workers=args.num_workers,
    )

    torch.manual_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(config.checkpoint_dir, "training_log.jsonl")

    # -- Pipeline & data ----------------------------------------------------
    print("Initializing pipeline...")
    pipeline = L2LPipeline(config)

    print(f"Loading dataset from {config.data_path} ...")
    dataset = PromptDataset(config.data_path)
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    print(f"  {train_size} train / {val_size} val prompts")

    collate_fn = make_collate_fn(pipeline.tokenizer, config.max_prompt_tokens)
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=config.num_workers,
    )
    # Stash val_loader on pipeline for mid-epoch eval access
    pipeline._val_loader = val_loader

    # -- Stage 1 ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 1: Train reinitialized layers only")
    print("=" * 60)
    pipeline.configure_stage1()

    trainable = sum(p.numel() for p in pipeline.middle_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pipeline.middle_model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    param_groups = pipeline.get_param_groups(stage=1)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_s1 = steps_per_epoch * config.stage1_epochs
    warmup_s1 = int(total_s1 * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_s1, total_s1)

    global_step = 0
    for epoch in range(config.stage1_epochs):
        global_step = train_one_epoch(
            pipeline, train_loader, optimizer, scheduler,
            stage=1, epoch=epoch, global_step=global_step,
            temperature_fn=config.stage1_temperature,
            log_path=log_path, config=config,
        )
        val_loss = validate(pipeline, val_loader, config.stage1_temperature)
        print(f"  Stage 1 Epoch {epoch} val_loss: {val_loss:.4f}")
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "step": global_step, "epoch": epoch,
                "stage": "stage1_val", "val_loss": round(val_loss, 6),
            }) + "\n")

    save_checkpoint(
        pipeline, optimizer, scheduler, 1, global_step, epoch,
        config.stage1_temperature,
        os.path.join(config.checkpoint_dir, "stage1_final.pt"),
    )

    # -- Stage 2 ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: Full finetune + temperature annealing")
    print("=" * 60)
    pipeline.configure_stage2()

    trainable = sum(p.numel() for p in pipeline.middle_model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    param_groups = pipeline.get_param_groups(stage=2)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    total_s2 = steps_per_epoch * config.stage2_epochs
    warmup_s2 = int(total_s2 * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_s2, total_s2)

    def temp_fn(step):
        return get_temperature(
            step, total_s2, config.temp_schedule,
            config.stage2_temp_start, config.stage2_temp_end,
        )

    s2_step = 0
    for epoch in range(config.stage2_epochs):
        s2_step = train_one_epoch(
            pipeline, train_loader, optimizer, scheduler,
            stage=2, epoch=epoch, global_step=s2_step,
            temperature_fn=temp_fn,
            log_path=log_path, config=config,
        )
        cur_temp = temp_fn(s2_step)
        val_loss = validate(pipeline, val_loader, cur_temp)
        print(f"  Stage 2 Epoch {epoch} val_loss: {val_loss:.4f}, temp: {cur_temp:.4f}")
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "step": s2_step, "epoch": epoch,
                "stage": "stage2_val", "val_loss": round(val_loss, 6),
                "temperature": round(cur_temp, 5),
            }) + "\n")

    # -- Save final model ---------------------------------------------------
    final_path = os.path.join(config.checkpoint_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    pipeline.middle_model.save_pretrained(final_path)
    pipeline.tokenizer.save_pretrained(final_path)
    print(f"\nFinal model saved to {final_path}")

    save_checkpoint(
        pipeline, optimizer, scheduler, 2, s2_step, epoch,
        cur_temp,
        os.path.join(config.checkpoint_dir, "stage2_final.pt"),
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
