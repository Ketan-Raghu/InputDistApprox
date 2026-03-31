"""
Qwen2.5-7B-Instruct logit extraction.

Runs inference on prompts from data/merged_prompts.jsonl in batches,
streaming the full logit vector (vocab_size=152064) for each prompt
to a temporary buffer file on disk. Each prompt's logits are written
as a single row then flushed, so memory stays bounded regardless of
dataset size.

Output format (binary, per-prompt):
  [4 bytes: prompt_index as int32]
  [4 bytes: vocab_size as int32]
  [vocab_size * 2 bytes: logits as float16]

Usage:
  python qwen_logit_inference.py [--batch-size 16] [--output /tmp/qwen_logits.buf]
"""

import argparse
import json
import os
import struct
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_PATH = "/home/ketan/LLMs/models/Qwen_Qwen2.5-7B-Instruct/"
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "data", "merged_prompts.jsonl")

PROMPT_TEMPLATE = (
    "You are a helpful assistant\n\n"
    "<PROMPT>{prompt}</PROMPT>\n\n"
    "<RESPONSE>"
)


def load_prompts(path: str) -> list[dict]:
    """Load all prompts from the JSONL file."""
    prompts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def format_prompt(raw_prompt: str) -> str:
    return PROMPT_TEMPLATE.format(prompt=raw_prompt)


def stream_logits_to_buffer(
    model,
    tokenizer,
    prompts: list[dict],
    batch_size: int,
    output_path: str,
    max_prompt_tokens: int = 2048,
):
    """
    Run batched inference and stream each prompt's full logit vector to a
    binary buffer file on disk.

    For each prompt we take the logits at the *last* token position (i.e. the
    distribution over the next token the model would generate).  Logits are
    stored as float16 to halve I/O and disk usage (~300 KB per prompt instead
    of ~600 KB).
    """
    vocab_size = model.config.vocab_size
    total = len(prompts)

    with open(output_path, "wb") as buf:
        for batch_start in tqdm(range(0, total, batch_size), desc="Batches"):
            batch_end = min(batch_start + batch_size, total)
            batch = prompts[batch_start:batch_end]

            texts = [format_prompt(p["prompt"]) for p in batch]

            encodings = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**encodings)

            # outputs.logits shape: (batch, seq_len, vocab_size)
            # For each item, find the last non-padding token position
            attention_mask = encodings["attention_mask"]  # (batch, seq_len)
            seq_lengths = attention_mask.sum(dim=1) - 1  # index of last real token

            for i in range(len(batch)):
                prompt_index = batch_start + i
                last_pos = seq_lengths[i].item()

                # Extract logit vector at the last token position
                logit_vec = outputs.logits[i, last_pos, :].cpu().to(torch.float16)
                assert logit_vec.shape[0] == vocab_size

                # Write: [prompt_index (i32)] [vocab_size (i32)] [logits (f16)]
                header = struct.pack("<ii", prompt_index, vocab_size)
                buf.write(header)
                buf.write(logit_vec.numpy().tobytes())
                buf.flush()

            # Free GPU memory for this batch
            del outputs, encodings
            torch.cuda.empty_cache()

    return vocab_size


def main():
    parser = argparse.ArgumentParser(
        description="Extract full logit vectors from Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Number of prompts per forward pass (default: 16)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for the output buffer file (default: tempfile in /tmp)",
    )
    parser.add_argument(
        "--max-prompt-tokens", type=int, default=2048,
        help="Maximum tokens per prompt after tokenization (default: 2048)",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        tempfile.gettempdir(), "qwen_logits.buf"
    )

    print(f"Loading prompts from {PROMPTS_PATH} ...")
    prompts = load_prompts(PROMPTS_PATH)
    print(f"  {len(prompts)} prompts loaded")

    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Streaming logits to {output_path}")
    print(f"  batch_size={args.batch_size}, max_prompt_tokens={args.max_prompt_tokens}")
    vocab_size = stream_logits_to_buffer(
        model, tokenizer, prompts,
        batch_size=args.batch_size,
        output_path=output_path,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    file_size_gb = os.path.getsize(output_path) / (1024 ** 3)
    print(f"Done. Output: {output_path} ({file_size_gb:.2f} GB)")
    print(f"  {len(prompts)} prompts x {vocab_size} vocab = "
          f"{len(prompts) * vocab_size * 2 / (1024**3):.2f} GB of logit data")


if __name__ == "__main__":
    main()
