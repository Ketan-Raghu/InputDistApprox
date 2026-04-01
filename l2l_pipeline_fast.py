"""
Optimized L2L Training Pipeline — Multi-GPU hardware-accelerated version.

Requires 3+ GPUs. Assigns one model per dedicated GPU for maximum throughput:
  GPU 0: Model A (frozen, autoregressive generation)
  GPU 1: Middle Model (trainable)
  GPU 2: Model B (frozen, reconstruction)

Key optimizations over l2l_pipeline.py:
  - Dedicated GPU per model (eliminates device_map="auto" cross-GPU fragmentation)
  - Separate Model A / Model B instances (enables pipeline parallelism)
  - Async Model A prefetch (overlaps next-batch generation with current backward)
  - TF32 matmul acceleration
  - Flash Attention 2 (with SDPA fallback)
  - Fused AdamW optimizer
  - Cached base embedding on middle GPU (avoids transferring large prob tensors)
  - Vectorized soft sequence construction (no Python for-loop over batch)
  - Larger default batch size (16 vs 2)
  - Persistent DataLoader workers with prefetching

Usage:
  python l2l_pipeline_fast.py [--batch-size 16] [--stage1-epochs 1] [--stage2-epochs 3]
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
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
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "l2l_fast")

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
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
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
    stage2_temp_end: float = 0.00
    temp_schedule: str = "cosine"
    val_split: float = 0.02
    log_every: int = 10
    eval_every_steps: int = 500
    save_every_steps: int = 1000
    seed: int = 42
    gradient_checkpointing: bool = True
    num_workers: int = 8
    gpu_a: int = 0
    gpu_mid: int = 1
    gpu_b: int = 2
    async_prefetch: bool = True
    compile_models: bool = False
    prefetch_factor: int = 4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PromptDataset(Dataset):
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
    def collate_fn(batch: list[str]):
        messages_list = [[{"role": "user", "content": p}] for p in batch]
        texts = [
            tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True,
            )
            for m in messages_list
        ]
        return tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_prompt_tokens,
        )
    return collate_fn


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class FastL2LPipeline(nn.Module):
    """
    Hardware-optimized L2L pipeline with dedicated GPU assignment.

    Three separate model instances avoid device_map="auto" fragmentation:
      Model A (GPU 0) — frozen, autoregressive generation
      Middle  (GPU 1) — trainable, reinitialized embed_tokens + layers 20-27
      Model B (GPU 2) — frozen, reconstruction for loss

    Gradient flow:
      loss (GPU 2) -> Model B activations -> .to() -> middle_probs (GPU 1)
      -> Middle model layers -> embed_tokens.weight
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.dtype = torch.bfloat16
        self._setup_hardware()
        self._load_models()
        if config.async_prefetch:
            self._executor = ThreadPoolExecutor(max_workers=1)
            self._gen_future = None

    @staticmethod
    def _to(tensor, device):
        """Transfer tensor to device via CPU.

        Direct GPU-to-GPU P2P transfers silently produce zeros on some
        multi-GPU systems. Routing through pinned CPU memory avoids this.
        """
        if tensor.device == device:
            return tensor
        return tensor.cpu().to(device)

    # -- setup --------------------------------------------------------------

    def _setup_hardware(self):
        n = torch.cuda.device_count()
        assert n >= 3, (
            f"Need 3+ GPUs, found {n}. Use l2l_pipeline.py for fewer GPUs."
        )
        # TF32 for faster matmuls on Ampere+ / Blackwell
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        cfg = self.config
        self.gpu_a = torch.device(f"cuda:{cfg.gpu_a}")
        self.gpu_mid = torch.device(f"cuda:{cfg.gpu_mid}")
        self.gpu_b = torch.device(f"cuda:{cfg.gpu_b}")
        # Separate CUDA stream for async Model A generation
        self.gen_stream = torch.cuda.Stream(device=self.gpu_a)

    def _detect_attn_impl(self):
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            return "sdpa"

    def _load_models(self):
        cfg = self.config
        attn_impl = self._detect_attn_impl()
        print(f"Attention: {attn_impl}")

        # Model A — frozen, generation on dedicated GPU
        print(f"Loading Model A on cuda:{cfg.gpu_a}...")
        self.model_a = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, dtype=self.dtype,
            device_map={"": cfg.gpu_a},
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        self.model_a.eval()
        for p in self.model_a.parameters():
            p.requires_grad = False

        # Middle model — trainable on dedicated GPU
        print(f"Loading Middle model on cuda:{cfg.gpu_mid}...")
        self.middle_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, dtype=self.dtype,
            device_map={"": cfg.gpu_mid},
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        self._reinit_middle()
        for p in self.middle_model.lm_head.parameters():
            p.requires_grad = False
        if cfg.gradient_checkpointing:
            self.middle_model.gradient_checkpointing_enable()

        # Model B — frozen, reconstruction on dedicated GPU
        print(f"Loading Model B on cuda:{cfg.gpu_b}...")
        self.model_b = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, dtype=self.dtype,
            device_map={"": cfg.gpu_b},
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        self.model_b.eval()
        for p in self.model_b.parameters():
            p.requires_grad = False

        # Cache Model B's embedding matrix on GPU 1 so the large
        # (B, S, 152064) middle_probs tensor stays on GPU 1 for the matmul.
        # Only the small (B, S, 3584) weighted_embeds crosses to GPU 2.
        # Route via CPU to avoid P2P zero-out.
        self._base_embed_cache = self._to(
            self.model_b.model.embed_tokens.weight.detach().clone(), self.gpu_mid,
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if cfg.compile_models:
            print("Compiling Middle model and Model B with torch.compile...")
            self.middle_model = torch.compile(self.middle_model)
            self.model_b = torch.compile(self.model_b)

        self._print_gpu_report()

    def _reinit_middle(self):
        nn.init.normal_(
            self.middle_model.model.embed_tokens.weight,
            mean=0.0, std=INITIALIZER_RANGE,
        )
        start = NUM_LAYERS - NUM_REINIT_LAYERS
        print(f"  Reinitializing embed_tokens + layers {start}-{NUM_LAYERS - 1}")
        for idx in range(start, NUM_LAYERS):
            reinit_module(self.middle_model.model.layers[idx], INITIALIZER_RANGE)

    def _get_saveable_middle(self):
        """Unwrap torch.compile wrapper if present."""
        m = self.middle_model
        return m._orig_mod if hasattr(m, "_orig_mod") else m

    def _print_gpu_report(self):
        num_gpus = torch.cuda.device_count()
        roles = {
            self.config.gpu_a: "Model A",
            self.config.gpu_mid: "Middle",
            self.config.gpu_b: "Model B",
        }
        print(f"\n{'=' * 50}")
        print(f"GPU Report ({num_gpus} devices)")
        print(f"{'=' * 50}")
        total_alloc = 0.0
        for i in range(num_gpus):
            alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            total_alloc += alloc
            role = f" [{roles[i]}]" if i in roles else ""
            print(f"  GPU {i}{role}: {alloc:.1f} / {total_gb:.0f} GB")
        print(f"  Total allocated: {total_alloc:.1f} GB")
        for name, m in [("Model A", self.model_a), ("Middle", self._get_saveable_middle()), ("Model B", self.model_b)]:
            sz = sum(p.numel() * p.element_size() for p in m.parameters()) / (1024 ** 3)
            print(f"  {name} weights: {sz:.2f} GB")
        cache_sz = self._base_embed_cache.numel() * self._base_embed_cache.element_size() / (1024 ** 3)
        print(f"  Base embed cache (GPU {self.config.gpu_mid}): {cache_sz:.2f} GB")
        print(f"  TF32: enabled | Async prefetch: {self.config.async_prefetch}")
        print(f"{'=' * 50}\n")

    # -- stage configuration ------------------------------------------------

    def configure_stage1(self):
        """Stage 1: only train reinitialized params (embed_tokens + layers 20-27)."""
        for p in self.middle_model.parameters():
            p.requires_grad = False
        for p in self._get_saveable_middle().model.embed_tokens.parameters():
            p.requires_grad = True
        start = NUM_LAYERS - NUM_REINIT_LAYERS
        for idx in range(start, NUM_LAYERS):
            for p in self._get_saveable_middle().model.layers[idx].parameters():
                p.requires_grad = True

    def configure_stage2(self):
        """Stage 2: unfreeze all middle model params except lm_head."""
        for p in self.middle_model.parameters():
            p.requires_grad = True
        for p in self._get_saveable_middle().lm_head.parameters():
            p.requires_grad = False

    def get_param_groups(self, stage: int) -> list[dict]:
        cfg = self.config
        start = NUM_LAYERS - NUM_REINIT_LAYERS
        if stage == 1:
            params = [p for p in self.middle_model.parameters() if p.requires_grad]
            return [{"params": params, "lr": cfg.stage1_lr}]
        reinit_params, pretrained_params = [], []
        for name, p in self._get_saveable_middle().named_parameters():
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

    # -- generation & soft sequence -----------------------------------------

    @torch.no_grad()
    def generate_model_a(self, input_ids, attention_mask):
        """Autoregressive generation on Model A's dedicated GPU."""
        input_ids = input_ids.to(self.gpu_a)
        attention_mask = attention_mask.to(self.gpu_a)

        gen_output = self.model_a.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )

        raw_logits = torch.stack(gen_output.logits, dim=1)  # (B, gen, V)
        prompt_len = input_ids.shape[1]
        gen_tokens = gen_output.sequences[:, prompt_len:]
        is_eos = gen_tokens == IM_END_ID
        has_eos = is_eos.any(dim=1)
        first_eos = is_eos.int().argmax(dim=1)
        gen_lengths = torch.where(
            has_eos, first_eos,
            torch.full_like(first_eos, gen_tokens.shape[1]),
        ).clamp(min=1)

        return raw_logits, gen_lengths

    def build_soft_sequence(self, raw_logits, gen_lengths, temperature):
        """
        Vectorized soft sequence construction — no Python for-loop over batch.

        Builds: [<|im_start|>, assistant, *prob_vectors, <|im_start|>, user]
        with per-item padding and attention mask.
        """
        B = raw_logits.shape[0]
        device = raw_logits.device
        temp = max(temperature, 1e-6)

        probs = F.softmax(raw_logits / temp, dim=-1)
        max_gen = gen_lengths.max().item()
        max_total = max_gen + 4  # 2 prefix + gen + 2 suffix

        output = torch.zeros(B, max_total, VOCAB_SIZE, device=device, dtype=self.dtype)

        # Prefix (positions 0-1, same for all items)
        output[:, 0, IM_START_ID] = 1.0
        output[:, 1, ASSISTANT_ID] = 1.0

        # Prob vectors (position 2..2+gen, masked per item)
        gen_idx = torch.arange(max_gen, device=device).unsqueeze(0)  # (1, max_gen)
        gen_valid = gen_idx < gen_lengths.unsqueeze(1)  # (B, max_gen)
        output[:, 2:2 + max_gen] = probs[:, :max_gen] * gen_valid.unsqueeze(-1)

        # Suffix (per-item variable position)
        batch_idx = torch.arange(B, device=device)
        suffix_start = gen_lengths + 2
        output[batch_idx, suffix_start, IM_START_ID] = 1.0
        output[batch_idx, suffix_start + 1, USER_ID] = 1.0

        # Attention mask
        total_len = suffix_start + 2
        pos_idx = torch.arange(max_total, device=device).unsqueeze(0)
        mask = (pos_idx < total_len.unsqueeze(1)).to(self.dtype)

        return output, mask

    def prepare_batch(self, input_ids, attention_mask, temperature):
        """Generate Model A output and build soft sequence (synchronous)."""
        raw_logits, gen_lengths = self.generate_model_a(input_ids, attention_mask)
        prob_vectors, mask = self.build_soft_sequence(
            raw_logits, gen_lengths, temperature,
        )
        prob_vectors = prob_vectors.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        return prob_vectors, mask

    # -- async prefetch -----------------------------------------------------

    def submit_generation(self, input_ids, attention_mask, temperature):
        """Start Model A generation for the next batch in a background thread.

        Runs on a separate CUDA stream (gpu_a) so it overlaps with the
        training forward/backward on gpu_mid + gpu_b.
        """
        def _gen():
            with torch.cuda.stream(self.gen_stream):
                result = self.prepare_batch(input_ids, attention_mask, temperature)
            self.gen_stream.synchronize()
            return result
        self._gen_future = self._executor.submit(_gen)

    def collect_generation(self):
        """Block until the async generation completes and return (prob_vectors, mask)."""
        result = self._gen_future.result()
        self._gen_future = None
        return result

    # -- training forward ---------------------------------------------------

    def train_forward(self, prob_vectors, mask):
        """
        Differentiable forward through Middle Model + Model B -> loss.

        prob_vectors / mask are on gpu_a (Model A's device). All cross-GPU
        transfers route via CPU to avoid P2P zero-out issues.
        """
        # -- Middle model (GPU 1) --
        prob_mid = self._to(prob_vectors, self.gpu_mid)
        mask_mid = self._to(mask, self.gpu_mid)

        mid_embed = self._get_saveable_middle().model.embed_tokens.weight
        soft_embeds = prob_mid @ mid_embed  # (B, S, 3584)

        middle_out = self.middle_model(
            inputs_embeds=soft_embeds,
            attention_mask=mask_mid,
            use_cache=False,
        )
        # Stabilize: reinitialized layers can produce extreme logits
        middle_logits = middle_out.logits.float().clamp(-65504, 65504)
        middle_probs = F.softmax(middle_logits, dim=-1).to(self.dtype)
        middle_probs = middle_probs.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)

        # Compute weighted_embeds on GPU 1 using cached base embedding
        # so the large middle_probs (B, S, 152064) stays local
        weighted_embeds = middle_probs @ self._base_embed_cache  # (B, S, 3584)

        # -- Model B (GPU 2) — only (B, S, 3584) crosses GPUs --
        model_b_out = self.model_b(
            inputs_embeds=self._to(weighted_embeds, self.gpu_b),
            attention_mask=self._to(mask, self.gpu_b),
            use_cache=False,
        )

        # -- Shifted autoregressive soft cross-entropy (GPU 2) --
        b_logits = model_b_out.logits[:, :-1, :]
        targets = self._to(prob_vectors[:, 1:, :], self.gpu_b)
        shifted_mask = self._to(mask[:, 1:], self.gpu_b)

        log_probs = F.log_softmax(b_logits.float(), dim=-1)
        per_token_loss = -(targets.float() * log_probs).nan_to_num(0.0).sum(dim=-1)
        loss = (per_token_loss * shifted_mask).sum() / shifted_mask.sum().clamp(min=1)

        return loss

    def forward(self, input_ids, attention_mask, temperature):
        """Combined generate + train_forward (for validation / compatibility)."""
        prob_vectors, mask = self.prepare_batch(input_ids, attention_mask, temperature)
        return self.train_forward(prob_vectors, mask)


# ---------------------------------------------------------------------------
# Prefetch iterator
# ---------------------------------------------------------------------------
class PrefetchIterator:
    """
    Wraps a dataloader to prefetch Model A generations one batch ahead.

    While the main thread runs forward/backward on GPU 1+2, the background
    thread generates the next batch's prob_vectors on GPU 0 in parallel.
    """

    def __init__(self, pipeline, dataloader, temperature):
        self.pipeline = pipeline
        self.dataloader = dataloader
        self.temperature = temperature

    def set_temperature(self, t):
        self.temperature = t

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        it = iter(self.dataloader)

        # First batch: generate synchronously (nothing to overlap with)
        try:
            first = next(it)
        except StopIteration:
            return

        current = self.pipeline.prepare_batch(
            first["input_ids"], first["attention_mask"], self.temperature,
        )

        for nxt in it:
            # Start next batch generation in background
            self.pipeline.submit_generation(
                nxt["input_ids"], nxt["attention_mask"], self.temperature,
            )
            # Yield current batch for training (runs on GPU 1+2)
            yield current
            # Collect finished background generation
            current = self.pipeline.collect_generation()

        # Yield the last batch
        yield current


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def get_temperature(step, total_steps, schedule="cosine", start=1.0, end=0.05):
    if total_steps <= 0:
        return start
    progress = min(step / total_steps, 1.0)
    if schedule == "cosine":
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
    return start + (end - start) * progress


def save_checkpoint(pipeline, optimizer, scheduler, stage, step, epoch, temp, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "middle_model_state_dict": pipeline._get_saveable_middle().state_dict(),
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


def load_checkpoint(path, pipeline, optimizer=None, scheduler=None):
    """
    Load a checkpoint. Returns dict with 'stage', 'step', 'epoch', 'temperature'.

    Restores middle model weights. Optionally restores optimizer/scheduler
    state if provided (only meaningful when resuming the same stage).
    """
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Load middle model weights (handles cross-device mapping automatically)
    pipeline._get_saveable_middle().load_state_dict(ckpt["middle_model_state_dict"])
    print(f"  Restored middle model weights (stage {ckpt['stage']}, step {ckpt['step']})")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"  Restored optimizer state")

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"  Restored scheduler state")

    return {
        "stage": ckpt["stage"],
        "step": ckpt["step"],
        "epoch": ckpt["epoch"],
        "temperature": ckpt.get("temperature", 1.0),
    }


def train_one_epoch(
    pipeline, dataloader, optimizer, scheduler,
    *, stage, epoch, global_step, temperature_fn, log_path, config,
):
    """Train for one epoch with optional async prefetching."""
    pipeline.middle_model.train()
    running_loss = 0.0
    num_losses = 0
    optimizer.zero_grad()

    temp = temperature_fn(global_step) if callable(temperature_fn) else temperature_fn

    if config.async_prefetch:
        batch_source = PrefetchIterator(pipeline, dataloader, temp)
    else:
        def _sync():
            for batch in dataloader:
                t = temperature_fn(global_step) if callable(temperature_fn) else temperature_fn
                yield pipeline.prepare_batch(
                    batch["input_ids"], batch["attention_mask"], t,
                )
        batch_source = _sync()

    pbar = tqdm(
        enumerate(batch_source), total=len(dataloader),
        desc=f"Stage {stage} Epoch {epoch}",
    )
    t_log = time.perf_counter()

    for batch_idx, (prob_vectors, mask) in pbar:
        temp = temperature_fn(global_step) if callable(temperature_fn) else temperature_fn
        if config.async_prefetch and hasattr(batch_source, "set_temperature"):
            batch_source.set_temperature(temp)

        loss = pipeline.train_forward(prob_vectors, mask)
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
                now = time.perf_counter()
                elapsed = now - t_log
                sec_per_step = elapsed / config.log_every
                tokens_per_sec = (
                    config.batch_size
                    * config.gradient_accumulation_steps
                    * mask.shape[1]
                ) / sec_per_step
                t_log = now

                avg_loss = running_loss / max(num_losses, 1)
                lr = optimizer.param_groups[0]["lr"]
                gpu_mid = torch.cuda.memory_allocated(config.gpu_mid) / (1024 ** 3)
                gpu_b = torch.cuda.memory_allocated(config.gpu_b) / (1024 ** 3)

                entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "stage": f"stage{stage}",
                    "loss": round(loss.item() * config.gradient_accumulation_steps, 6),
                    "avg_loss": round(avg_loss, 6),
                    "lr": lr,
                    "grad_norm": round(grad_norm, 4),
                    "gpu_mid_gb": round(gpu_mid, 2),
                    "gpu_b_gb": round(gpu_b, 2),
                    "tokens_per_sec": round(tokens_per_sec, 1),
                    "temperature": round(temp, 5),
                    "sec_per_step": round(sec_per_step, 2),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    t=f"{temp:.3f}",
                    tok=f"{tokens_per_sec:.0f}/s",
                    gn=f"{grad_norm:.2f}",
                )

            # Periodic checkpoint
            if global_step % config.save_every_steps == 0:
                ckpt = os.path.join(
                    config.checkpoint_dir,
                    f"stage{stage}_step{global_step}.pt",
                )
                save_checkpoint(
                    pipeline, optimizer, scheduler,
                    stage, global_step, epoch, temp, ckpt,
                )

            # Periodic eval
            if global_step % config.eval_every_steps == 0:
                val_loss = validate(pipeline, pipeline._val_loader, temp)
                print(f"  Step {global_step} val_loss: {val_loss:.4f}")
                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "step": global_step,
                        "epoch": epoch,
                        "stage": f"stage{stage}_val",
                        "val_loss": round(val_loss, 6),
                        "temperature": round(temp, 5),
                    }) + "\n")
                pipeline.middle_model.train()

    return global_step


@torch.no_grad()
def validate(pipeline, dataloader, temperature, max_batches=50):
    """Run validation (synchronous, no prefetch needed)."""
    pipeline.middle_model.eval()
    total_loss = 0.0
    count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
        if batch_idx >= max_batches:
            break
        prob_vectors, mask = pipeline.prepare_batch(
            batch["input_ids"], batch["attention_mask"], temperature,
        )
        loss = pipeline.train_forward(prob_vectors, mask)
        if not torch.isnan(loss):
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimized L2L Training Pipeline (multi-GPU)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
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
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu-a", type=int, default=0, help="GPU for Model A")
    parser.add_argument("--gpu-mid", type=int, default=1, help="GPU for Middle model")
    parser.add_argument("--gpu-b", type=int, default=2, help="GPU for Model B")
    parser.add_argument("--async-prefetch", action="store_true", default=True)
    parser.add_argument("--no-async-prefetch", dest="async_prefetch", action="store_false")
    parser.add_argument("--compile", dest="compile_models", action="store_true", default=False)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from (skips completed stages)",
    )
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
        gpu_a=args.gpu_a,
        gpu_mid=args.gpu_mid,
        gpu_b=args.gpu_b,
        async_prefetch=args.async_prefetch,
        compile_models=args.compile_models,
        prefetch_factor=args.prefetch_factor,
    )

    torch.manual_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(config.checkpoint_dir, "training_log.jsonl")

    # -- Pipeline & data ----------------------------------------------------
    print("Initializing optimized pipeline...")
    pipeline = FastL2LPipeline(config)

    print(f"Loading dataset from {config.data_path}...")
    dataset = PromptDataset(config.data_path)
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    print(f"  {train_size} train / {val_size} val prompts")

    collate_fn = make_collate_fn(pipeline.tokenizer, config.max_prompt_tokens)
    worker_kwargs = {}
    if config.num_workers > 0:
        worker_kwargs = {
            "persistent_workers": True,
            "prefetch_factor": config.prefetch_factor,
        }
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=config.num_workers,
        pin_memory=True, **worker_kwargs,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=config.num_workers,
        **worker_kwargs,
    )
    pipeline._val_loader = val_loader

    effective_batch = config.batch_size * config.gradient_accumulation_steps
    print(f"  Effective batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {effective_batch}")

    # -- Resume handling ----------------------------------------------------
    resume_stage = 0  # 0 = start fresh, 1 = stage 1 done, 2 = stage 2 mid
    resume_step = 0
    resume_epoch = 0
    if args.resume:
        ckpt_info = load_checkpoint(args.resume, pipeline)
        resume_stage = ckpt_info["stage"]
        resume_step = ckpt_info["step"]
        resume_epoch = ckpt_info["epoch"]
        print(f"  Resumed from stage {resume_stage}, step {resume_step}, epoch {resume_epoch}")

    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_params = sum(p.numel() for p in pipeline.middle_model.parameters())

    # -- Stage 1 ------------------------------------------------------------
    if resume_stage < 1:
        print(f"\n{'=' * 60}")
        print("STAGE 1: Train reinitialized layers only")
        print(f"{'=' * 60}")
        pipeline.configure_stage1()

        trainable = sum(p.numel() for p in pipeline.middle_model.parameters() if p.requires_grad)
        print(f"  Trainable: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.1f}%)")

        param_groups = pipeline.get_param_groups(stage=1)
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=config.weight_decay, fused=True,
        )
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
    else:
        print(f"\n  Skipping Stage 1 (already completed in checkpoint)")

    # -- Stage 2 ------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STAGE 2: Full finetune + temperature annealing")
    print(f"{'=' * 60}")
    pipeline.configure_stage2()

    trainable = sum(p.numel() for p in pipeline.middle_model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.1f}%)")

    param_groups = pipeline.get_param_groups(stage=2)
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=config.weight_decay, fused=True,
    )
    total_s2 = steps_per_epoch * config.stage2_epochs
    warmup_s2 = int(total_s2 * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_s2, total_s2)

    def temp_fn(step):
        return get_temperature(
            step, total_s2, config.temp_schedule,
            config.stage2_temp_start, config.stage2_temp_end,
        )

    # Resume mid-stage-2 if applicable
    s2_step = 0
    s2_start_epoch = 0
    if resume_stage == 2:
        s2_step = resume_step
        s2_start_epoch = resume_epoch
        # Restore optimizer/scheduler for same-stage resume
        load_checkpoint(args.resume, pipeline, optimizer, scheduler)
        print(f"  Resuming Stage 2 from step {s2_step}, epoch {s2_start_epoch}")

    for epoch in range(s2_start_epoch, config.stage2_epochs):
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
    pipeline._get_saveable_middle().save_pretrained(final_path)
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
