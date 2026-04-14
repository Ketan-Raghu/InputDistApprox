"""
T2T Output Generation — 5-GPU parallel output generation for inverse prediction training.

Loads Qwen2.5-7B-Instruct on each of 5 GPUs with different seeds, generates one
output per prompt per GPU. Each (output, prompt) pair becomes a training sample.
Results are merged into a single JSONL for training.

Usage:
  python generate_t2t_outputs.py [--num-gpus 5] [--batch-size 64]
  python generate_t2t_outputs.py --merge-only  # skip generation, just merge
"""

import argparse
import json
import os
import time

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_PATH = "/home/ketan/LLMs/models/Qwen_Qwen2.5-7B-Instruct/"
INPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "merged_prompts.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def load_prompts(path: str) -> list[dict]:
    """Load prompts from merged JSONL."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def count_lines(path: str) -> int:
    """Count lines in file, return 0 if file doesn't exist."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def generate_worker(gpu_id: int, args, prompts: list[dict]):
    """Worker function for a single GPU. Generates one output per prompt."""
    torch.cuda.set_device(gpu_id)
    seed = args.seed + gpu_id
    torch.manual_seed(seed)

    output_path = os.path.join(args.output_dir, f"t2t_gen_gpu{gpu_id}.jsonl")

    # Resume: skip already-generated prompts
    already_done = 0
    if args.resume:
        already_done = count_lines(output_path)
        if already_done >= len(prompts):
            print(f"[GPU {gpu_id}] Already complete ({already_done}/{len(prompts)}), skipping.")
            return
        if already_done > 0:
            print(f"[GPU {gpu_id}] Resuming from prompt {already_done}/{len(prompts)}")

    prompts_remaining = prompts[already_done:]

    # Load model
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    print(f"[GPU {gpu_id}] Loading model (seed={seed}, attn={attn_impl})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": gpu_id},
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generation config
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=1.05,
    )

    batch_size = args.batch_size
    max_tokens_hit = 0
    total_generated = 0
    file_mode = "a" if already_done > 0 else "w"

    pbar = tqdm(
        total=len(prompts_remaining),
        desc=f"GPU {gpu_id}",
        position=gpu_id,
        leave=True,
    )

    with open(output_path, file_mode) as outf:
        for batch_start in range(0, len(prompts_remaining), batch_size):
            batch_prompts = prompts_remaining[batch_start:batch_start + batch_size]

            # Build chat-templated inputs
            texts = []
            for entry in batch_prompts:
                messages = [
                    {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                    {"role": "user", "content": entry["prompt"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                texts.append(text)

            # Tokenize with left-padding for batched generation
            try:
                inputs = tokenizer(
                    texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=1024,
                ).to(f"cuda:{gpu_id}")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **gen_kwargs,
                    )
            except torch.cuda.OutOfMemoryError:
                # Halve batch size and retry
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                print(f"\n[GPU {gpu_id}] OOM! Reducing batch size to {batch_size}")
                # Re-process this batch with smaller size
                for sub_start in range(0, len(batch_prompts), batch_size):
                    sub_batch = batch_prompts[sub_start:sub_start + batch_size]
                    sub_texts = []
                    for entry in sub_batch:
                        messages = [
                            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                            {"role": "user", "content": entry["prompt"]},
                        ]
                        sub_texts.append(tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                        ))
                    sub_inputs = tokenizer(
                        sub_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=1024,
                    ).to(f"cuda:{gpu_id}")

                    with torch.no_grad():
                        sub_outputs = model.generate(**sub_inputs, **gen_kwargs)

                    prompt_lens = sub_inputs["input_ids"].shape[1]
                    for i, entry in enumerate(sub_batch):
                        gen_tokens = sub_outputs[i][prompt_lens:]
                        decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                        has_eos = any(
                            t.item() == tokenizer.eos_token_id
                            for t in gen_tokens
                        )
                        if not has_eos:
                            max_tokens_hit += 1
                        record = {
                            "prompt": entry["prompt"],
                            "source": entry.get("source", "unknown"),
                            "output": decoded,
                            "gpu_id": gpu_id,
                        }
                        outf.write(json.dumps(record) + "\n")
                        total_generated += 1
                    pbar.update(len(sub_batch))
                continue

            # Decode outputs — strip the input portion
            prompt_len = inputs["input_ids"].shape[1]
            for i, entry in enumerate(batch_prompts):
                gen_tokens = outputs[i][prompt_len:]
                decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                # Check if generation hit max_new_tokens (no EOS)
                has_eos = any(
                    t.item() == tokenizer.eos_token_id
                    for t in gen_tokens
                )
                if not has_eos:
                    max_tokens_hit += 1

                record = {
                    "prompt": entry["prompt"],
                    "source": entry.get("source", "unknown"),
                    "output": decoded,
                    "gpu_id": gpu_id,
                }
                outf.write(json.dumps(record) + "\n")
                total_generated += 1

            pbar.update(len(batch_prompts))
            outf.flush()

    pbar.close()
    print(
        f"\n[GPU {gpu_id}] Done. Generated {total_generated} outputs. "
        f"Max-tokens hit: {max_tokens_hit} ({100 * max_tokens_hit / max(total_generated, 1):.1f}%)"
    )


def merge_outputs(output_dir: str, num_gpus: int, expected_per_gpu: int):
    """Merge per-GPU output files into a single training pairs JSONL."""
    merged_path = os.path.join(output_dir, "t2t_training_pairs.jsonl")

    # Verify all files exist and have expected line counts
    gpu_files = []
    for gpu_id in range(num_gpus):
        path = os.path.join(output_dir, f"t2t_gen_gpu{gpu_id}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing GPU output file: {path}")
        n_lines = count_lines(path)
        if n_lines != expected_per_gpu:
            raise ValueError(
                f"GPU {gpu_id} file has {n_lines} lines, expected {expected_per_gpu}"
            )
        gpu_files.append(path)

    print(f"Merging {num_gpus} files ({expected_per_gpu} lines each) -> {merged_path}")

    # Open all files, interleave records
    handles = [open(p) for p in gpu_files]
    total = 0
    with open(merged_path, "w") as outf:
        for lines in zip(*handles):
            for line in lines:
                outf.write(line)
                total += 1

    for h in handles:
        h.close()

    print(f"Merged {total} training pairs to {merged_path}")
    return merged_path


def main():
    parser = argparse.ArgumentParser(description="T2T output generation (multi-GPU)")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--input-path", type=str, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--num-gpus", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--merge-only", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts
    print(f"Loading prompts from {args.input_path}...")
    prompts = load_prompts(args.input_path)
    print(f"  {len(prompts)} prompts loaded")

    if not args.merge_only:
        n_gpus = min(args.num_gpus, torch.cuda.device_count())
        if n_gpus < args.num_gpus:
            print(f"WARNING: Requested {args.num_gpus} GPUs but only {n_gpus} available")
        print(f"Spawning {n_gpus} workers (seeds {args.seed}..{args.seed + n_gpus - 1})")
        print(f"Each GPU generates 1 output per prompt -> {len(prompts)} outputs/GPU")
        print(f"Total training pairs: {n_gpus * len(prompts)}")

        t0 = time.time()
        mp.spawn(
            generate_worker,
            args=(args, prompts),
            nprocs=n_gpus,
            join=True,
        )
        elapsed = time.time() - t0
        print(f"\nGeneration complete in {elapsed / 60:.1f} minutes")

    # Merge
    merge_outputs(args.output_dir, args.num_gpus, len(prompts))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
