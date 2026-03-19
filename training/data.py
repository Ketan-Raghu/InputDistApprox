import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from transformers import PreTrainedTokenizer

from training.config import TrainingConfig

logger = logging.getLogger(__name__)


class FlippedConversationDataset(Dataset):
    """Dataset that flips ShareGPT conversations for inverse mapping training.

    Format: [BOS] gpt_response_1 [SEP] human_prompt_1 [SEP] gpt_response_2 [SEP] human_prompt_2 ... [EOS]
    Labels: -100 for gpt_response and separator tokens; only human_prompt tokens contribute to loss.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.examples = []

        raw_conversations = self._load_conversations(data_path)
        skipped = 0
        for conv in raw_conversations:
            result = self._process_conversation(conv)
            if result is not None:
                self.examples.append(result)
            else:
                skipped += 1

        logger.info(
            f"Loaded {len(self.examples)} conversations, skipped {skipped} "
            f"(empty gpt responses)"
        )

    def _load_conversations(self, data_path: str) -> List[dict]:
        conversations = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    conversations.append(json.loads(line))
        return conversations

    def _process_conversation(
        self, conv: dict
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build flipped input_ids and labels for a single conversation."""
        turns = conv["conversations"]

        # Pair up turns: (human, gpt) pairs
        segments = []  # list of (token_ids, is_target) tuples
        i = 0
        while i < len(turns):
            if turns[i]["from"] == "human" and i + 1 < len(turns) and turns[i + 1]["from"] == "gpt":
                human_text = turns[i]["value"].strip()
                gpt_text = turns[i + 1]["value"].strip()

                if not gpt_text:
                    return None  # Skip conversations with empty gpt responses

                # In flipped format: gpt first (input), then human (target)
                gpt_ids = self.tokenizer.encode(gpt_text, add_special_tokens=False)
                human_ids = self.tokenizer.encode(human_text, add_special_tokens=False)

                segments.append((gpt_ids, False))   # gpt response — masked
                segments.append((human_ids, True))   # human prompt — target
                i += 2
            else:
                i += 1

        if not segments:
            return None

        # Build sequence: [BOS] seg1 [SEP] seg2 [SEP] ... [EOS]
        input_ids = [self.config.bos_token_id]
        labels = [-100]  # BOS is masked

        for idx, (token_ids, is_target) in enumerate(segments):
            # Add separator before each segment (except the first)
            if idx > 0:
                input_ids.append(self.config.separator_token_id)
                labels.append(-100)  # Separator is masked

            input_ids.extend(token_ids)
            if is_target:
                labels.extend(token_ids)
            else:
                labels.extend([-100] * len(token_ids))

        # Add EOS
        input_ids.append(self.config.eos_token_id)
        labels.append(self.config.eos_token_id)  # EOS after last target is included in loss

        # Truncate to max_seq_length
        input_ids = input_ids[: self.config.max_seq_length]
        labels = labels[: self.config.max_seq_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Right-pad to longest sequence in batch."""
    max_len = max(ex["input_ids"].size(0) for ex in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for ex in batch:
        seq_len = ex["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([ex["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(
            torch.cat([ex["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def build_datasets(
    config: TrainingConfig, tokenizer: PreTrainedTokenizer
) -> Tuple[FlippedConversationDataset, FlippedConversationDataset]:
    """Build train/val split."""
    full_dataset = FlippedConversationDataset(config.data_path, tokenizer, config)

    val_size = int(len(full_dataset) * config.eval_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    logger.info(f"Train: {train_size}, Val: {val_size}")
    return train_dataset, val_dataset


def build_dataloader(
    dataset,
    config: TrainingConfig,
    batch_size: int,
    shuffle: bool = True,
    distributed: bool = False,
) -> DataLoader:
    """Build DataLoader with optional DistributedSampler for FSDP."""
    from functools import partial

    collate = partial(collate_fn, pad_token_id=config.pad_token_id)

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
