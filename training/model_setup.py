import logging
from typing import List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from training.config import TrainingConfig

logger = logging.getLogger(__name__)


def load_and_prepare_model(config: TrainingConfig):
    """Load model and tokenizer, configure pad token and gradient checkpointing."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = config.pad_token
    tokenizer.pad_token_id = config.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params / 1e9:.2f}B parameters")

    return model, tokenizer


def reinitialize_layers(
    model: nn.Module, layer_indices: List[int], std: float = 0.02
):
    """Reinitialize specified decoder layers with fresh random weights."""
    reinit_params = 0
    for idx in layer_indices:
        layer = model.model.layers[idx]
        for name, param in layer.named_parameters():
            if param.ndim >= 2:
                # Linear weights
                nn.init.normal_(param.data, mean=0.0, std=std)
            else:
                # RMSNorm weights (1-D)
                nn.init.ones_(param.data)
            reinit_params += param.numel()

    logger.info(
        f"Reinitialized layers {layer_indices}: "
        f"{reinit_params / 1e6:.1f}M parameters"
    )


def freeze_for_warmup(model: nn.Module, layer_indices: List[int]):
    """Freeze all parameters, then unfreeze only the specified layers."""
    for param in model.parameters():
        param.requires_grad = False

    unfrozen = 0
    for idx in layer_indices:
        for param in model.model.layers[idx].parameters():
            param.requires_grad = True
            unfrozen += param.numel()

    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Frozen for warmup: {unfrozen / 1e6:.1f}M trainable "
        f"({unfrozen / total * 100:.1f}% of {total / 1e9:.2f}B)"
    )


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters for full training."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"All parameters unfrozen: {trainable / 1e6:.1f}M trainable")


def _is_norm_weight(name: str) -> bool:
    """Check if a parameter is a normalization layer weight (no weight decay)."""
    return "norm" in name.lower() and "weight" in name.lower()


def build_optimizer_groups(
    model: nn.Module, config: TrainingConfig, stage: str
) -> torch.optim.AdamW:
    """Build optimizer with appropriate parameter groups for each stage."""
    reinit_layer_names = {f"model.layers.{i}." for i in config.reinit_layers}

    def _is_reinit_param(name: str) -> bool:
        return any(name.startswith(prefix) or f".{prefix}" in name for prefix in reinit_layer_names)

    if stage == "warmup":
        # Only reinitialized layers are trainable
        params_with_decay = []
        params_no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if _is_norm_weight(name):
                params_no_decay.append(param)
            else:
                params_with_decay.append(param)

        groups = [
            {"params": params_with_decay, "lr": config.stage1_lr, "weight_decay": config.weight_decay},
            {"params": params_no_decay, "lr": config.stage1_lr, "weight_decay": 0.0},
        ]

    elif stage == "full":
        # Differential learning rates
        pretrained_decay = []
        pretrained_no_decay = []
        reinit_decay = []
        reinit_no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_reinit = _is_reinit_param(name)
            is_norm = _is_norm_weight(name)

            if is_reinit and not is_norm:
                reinit_decay.append(param)
            elif is_reinit and is_norm:
                reinit_no_decay.append(param)
            elif not is_reinit and not is_norm:
                pretrained_decay.append(param)
            else:
                pretrained_no_decay.append(param)

        groups = [
            {"params": pretrained_decay, "lr": config.stage2_pretrained_lr, "weight_decay": config.weight_decay},
            {"params": pretrained_no_decay, "lr": config.stage2_pretrained_lr, "weight_decay": 0.0},
            {"params": reinit_decay, "lr": config.stage2_reinit_lr, "weight_decay": config.weight_decay},
            {"params": reinit_no_decay, "lr": config.stage2_reinit_lr, "weight_decay": 0.0},
        ]

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Filter out empty groups
    groups = [g for g in groups if len(g["params"]) > 0]

    optimizer = torch.optim.AdamW(
        groups,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )

    for i, g in enumerate(groups):
        logger.info(
            f"Optimizer group {i}: {sum(p.numel() for p in g['params']) / 1e6:.1f}M params, "
            f"lr={g['lr']}, wd={g['weight_decay']}"
        )

    return optimizer
