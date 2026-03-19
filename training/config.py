from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    # Paths
    model_path: str = "/home/ketan/LLMs/models/meta-llama_Llama-3.1-8B"
    data_path: str = "data/sharegpt_llama3.1_8b/sharegpt_llama3.1_8b.jsonl"
    output_dir: str = "checkpoints/inverse_mapping"

    # Tokenizer special tokens
    separator_token: str = "<|reserved_special_token_0|>"
    separator_token_id: int = 128002
    pad_token: str = "<|finetune_right_pad_id|>"
    pad_token_id: int = 128004
    bos_token_id: int = 128000
    eos_token_id: int = 128001

    # Model
    max_seq_length: int = 2048
    reinit_layers: List[int] = field(default_factory=lambda: [28, 29, 30, 31])
    initializer_range: float = 0.02

    # Stage 1: Warmup (DDP across GPUs, only reinitialized layers)
    stage1_epochs: int = 1
    stage1_lr: float = 2e-4
    stage1_per_gpu_batch: int = 4
    stage1_grad_accum: int = 4

    # Stage 2: Full training (multi-GPU FSDP, differential LR)
    stage2_epochs: int = 3
    stage2_pretrained_lr: float = 2e-5
    stage2_reinit_lr: float = 2e-4
    stage2_per_gpu_batch: int = 2
    stage2_grad_accum: int = 8

    # Shared optimizer / scheduler
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    seed: int = 42

    # Infrastructure
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    gradient_checkpointing: bool = True
    checkpoint_every: int = 500
    eval_split: float = 0.02

    # Logging
    log_every: int = 10
    
    log_file: str = "training_log.jsonl"
