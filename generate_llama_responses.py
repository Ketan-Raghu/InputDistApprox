"""
Generate LLaMA-3.1-8B responses for the ShareGPT dataset.

Replaces GPT-4 responses in shibing624/sharegpt_gpt4 (~103K multi-turn
conversations) with responses from the LLaMA-3.1-8B base model.

Uses 5 GPUs in parallel (GPUs 0, 2, 3, 4, 5 — skipping GPU 1 for display).
Supports checkpointing and resumption.
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# for i in range(torch.cuda.device_count()):
#         total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
#         allocated = torch.cuda.memory_allocated(i) / 1e9
#         print(f"GPU {i}: {allocated:.2f}GB / {total_memory:.2f}GB allocated")

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_PATH = "/home/ketan/LLMs/models/meta-llama_Llama-3.1-8B"
DATASET_NAME = "shibing624/sharegpt_gpt4"
OUTPUT_DIR = Path("data/sharegpt_llama3.1_8b")
GPU_IDS = [0, 1, 2, 3, 4]  # skip GPU 5 (display server)
NUM_GPUS = len(GPU_IDS)
MAX_NEW_TOKENS = 2048
CHECKPOINT_EVERY = 100
TEMPERATURE = 0.6
TOP_P = 0.9

# Maximum context length to feed the model (leave room for generation)
MAX_CONTEXT_TOKENS = 4096


# ─── Prompt formatting ──────────────────────────────────────────────────────

SYSTEM_PREAMBLE = "A conversation between a user and a helpful assistant.\n\n"


def format_prompt(conversation_so_far: list[dict], system_msg: str | None = None) -> str:
    """Build a text-completion prompt from conversation history.

    Args:
        conversation_so_far: List of {"from": "human"/"gpt", "value": "..."} dicts.
            The last entry must be a human turn (the one we want the model to respond to).
        system_msg: Optional system message to prepend.

    Returns:
        A string prompt ending with "Assistant:" for the model to complete.
    """
    parts = []
    if system_msg:
        parts.append(f"{system_msg}\n\n")
    parts.append(SYSTEM_PREAMBLE)

    for turn in conversation_so_far:
        role = "User" if turn["from"] == "human" else "Assistant"
        parts.append(f"{role}: {turn['value']}\n")

    parts.append("Assistant:")
    return "".join(parts)


# ─── Worker ──────────────────────────────────────────────────────────────────

def worker(gpu_id: int, shard: list[dict], worker_idx: int):
    """Process a shard of conversations on a single GPU.

    Each completed conversation is appended to a per-worker JSONL checkpoint
    file.  On resumption, already-completed indices are skipped.
    """
    device = torch.device(f"cuda:{gpu_id}")
    checkpoint_path = OUTPUT_DIR / f"checkpoint_gpu{gpu_id}.jsonl"

    # ── Determine resume point ───────────────────────────────────────────
    completed = 0
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = sum(1 for _ in f)
    if completed >= len(shard):
        print(f"[GPU {gpu_id}] Already completed all {len(shard)} conversations, skipping.")
        return
    if completed > 0:
        print(f"[GPU {gpu_id}] Resuming from conversation {completed}/{len(shard)}")

    # ── Load model & tokenizer ───────────────────────────────────────────
    print(f"[GPU {gpu_id}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    print(f"[GPU {gpu_id}] Model loaded.")

    # ── Process conversations ────────────────────────────────────────────
    ckpt_file = open(checkpoint_path, "a")
    try:
        for conv_idx in tqdm(
            range(completed, len(shard)),
            desc=f"GPU {gpu_id}",
            position=worker_idx,
            leave=True,
        ):
            conv = shard[conv_idx]
            original_turns = conv["conversations"]

            # Extract optional system message
            system_msg = None
            turns = []
            for t in original_turns:
                if t["from"] == "system":
                    system_msg = t["value"]
                else:
                    turns.append(t)

            # Build new conversation with LLaMA responses
            new_turns = []
            context: list[dict] = []  # accumulated context for prompting

            for turn in turns:
                if turn["from"] == "human":
                    context.append({"from": "human", "value": turn["value"]})
                    new_turns.append({"from": "human", "value": turn["value"]})

                    # Generate LLaMA response
                    prompt = format_prompt(context, system_msg)
                    input_ids = tokenizer.encode(
                        prompt, return_tensors="pt", truncation=True,
                        max_length=MAX_CONTEXT_TOKENS,
                    ).to(device)

                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids,
                            max_new_tokens=MAX_NEW_TOKENS,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    # Decode only the newly generated tokens
                    generated_ids = output_ids[0, input_ids.shape[1]:]
                    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                    # Stop at "User:" if the model generates further turns
                    if "\nUser:" in response:
                        response = response[:response.index("\nUser:")].strip()

                    new_turns.append({"from": "gpt", "value": response})
                    context.append({"from": "gpt", "value": response})

                # Skip original gpt/assistant turns (we replace them)

            result = {"conversations": new_turns}
            ckpt_file.write(json.dumps(result, ensure_ascii=False) + "\n")

            if (conv_idx + 1) % CHECKPOINT_EVERY == 0:
                ckpt_file.flush()

    finally:
        ckpt_file.close()

    print(f"[GPU {gpu_id}] Finished all {len(shard)} conversations.")


# ─── Main ────────────────────────────────────────────────────────────────────

def merge_checkpoints():
    """Merge per-worker JSONL files into a single output file and HF dataset."""
    output_jsonl = OUTPUT_DIR / "sharegpt_llama3.1_8b.jsonl"
    all_conversations = []

    for gpu_id in GPU_IDS:
        ckpt = OUTPUT_DIR / f"checkpoint_gpu{gpu_id}.jsonl"
        if not ckpt.exists():
            print(f"Warning: checkpoint for GPU {gpu_id} not found, skipping.")
            continue
        with open(ckpt) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_conversations.append(json.loads(line))

    print(f"Total conversations: {len(all_conversations)}")

    # Write merged JSONL
    with open(output_jsonl, "w") as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"Saved merged JSONL to {output_jsonl}")

    # Save as HuggingFace dataset
    hf_dir = OUTPUT_DIR / "hf_dataset"
    ds = Dataset.from_list(all_conversations)
    ds.save_to_disk(str(hf_dir))
    print(f"Saved HuggingFace dataset to {hf_dir}")


def main():
    mp.set_start_method("spawn", force=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME, split="train")
    data = list(ds)
    total = len(data)
    print(f"Dataset loaded: {total} conversations")

    # Split into shards
    shard_size = (total + NUM_GPUS - 1) // NUM_GPUS
    shards = []
    for i in range(NUM_GPUS):
        start = i * shard_size
        end = min(start + shard_size, total)
        shards.append(data[start:end])

    print(f"Sharded into {NUM_GPUS} parts: {[len(s) for s in shards]}")

    # Launch workers
    start_time = time.time()
    processes = []
    for worker_idx, (gpu_id, shard) in enumerate(zip(GPU_IDS, shards)):
        p = mp.Process(target=worker, args=(gpu_id, shard, worker_idx))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"\nGeneration complete in {elapsed / 3600:.1f} hours.")

    # Merge all checkpoint files
    merge_checkpoints()


if __name__ == "__main__":
    main()
