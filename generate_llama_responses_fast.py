"""
Fast batched inference for LLaMA-3.1-8B response generation on ShareGPT.

Instead of processing one conversation at a time per GPU, this script batches
multiple conversations together (batch_size=16 by default). This amortizes
the memory-bandwidth cost of reading model weights across all sequences in the
batch, yielding ~10-14x throughput improvement over the single-sequence version.

Uses 5 GPUs in parallel. Supports checkpointing and resumption.
"""

import json
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

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
BATCH_SIZE = 16
USE_COMPILE = False  # opt-in; may have issues with batched generate()

# Maximum context length to feed the model (leave room for generation)
MAX_CONTEXT_TOKENS = 4096

# ─── Prompt formatting ──────────────────────────────────────────────────────

SYSTEM_PREAMBLE = "A conversation between a user and a helpful assistant.\n\n"


def format_prompt(conversation_so_far: list[dict], system_msg: str | None = None) -> str:
    """Build a text-completion prompt from conversation history.

    The last entry in conversation_so_far must be a human turn.
    Returns a string ending with "Assistant:" for the model to complete.
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


# ─── Conversation state ─────────────────────────────────────────────────────

def init_conversation_states(shard: list[dict], completed_indices: set[int]):
    """Build a list of conversation states, skipping already-completed ones.

    Each state tracks the conversation's progress through its human turns.
    """
    states = []
    for i, conv in enumerate(shard):
        if i in completed_indices:
            continue

        original_turns = conv["conversations"]
        system_msg = None
        turns = []
        for t in original_turns:
            if t["from"] == "system":
                system_msg = t["value"]
            else:
                turns.append(t)

        human_turn_indices = [j for j, t in enumerate(turns) if t["from"] == "human"]
        if not human_turn_indices:
            continue

        states.append({
            "shard_idx": i,
            "turns": turns,
            "system_msg": system_msg,
            "human_turn_indices": human_turn_indices,
            "current_human_pos": 0,  # index into human_turn_indices
            "context": [],
            "new_turns": [],
        })

    # Sort by total human turns ascending — short conversations first
    states.sort(key=lambda s: len(s["human_turn_indices"]))
    return states


# ─── Batched worker ──────────────────────────────────────────────────────────

def batched_worker(gpu_id: int, shard: list[dict], worker_idx: int):
    """Process a shard of conversations on a single GPU using batched inference."""
    device = torch.device(f"cuda:{gpu_id}")
    checkpoint_path = OUTPUT_DIR / f"checkpoint_gpu{gpu_id}.jsonl"

    torch.set_float32_matmul_precision("medium")

    # ── Determine completed conversations ────────────────────────────────
    completed_indices: set[int] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if "_shard_idx" in record:
                        completed_indices.add(record["_shard_idx"])
    if completed_indices:
        print(f"[GPU {gpu_id}] Resuming — {len(completed_indices)} conversations already done")

    # ── Build conversation states ────────────────────────────────────────
    states = init_conversation_states(shard, completed_indices)
    total_to_process = len(states)
    if total_to_process == 0:
        print(f"[GPU {gpu_id}] All conversations already completed, skipping.")
        return
    print(f"[GPU {gpu_id}] {total_to_process} conversations to process")

    # ── Load model & tokenizer ───────────────────────────────────────────
    print(f"[GPU {gpu_id}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model.eval()

    if USE_COMPILE:
        torch._dynamo.config.cache_size_limit = 64
        model = torch.compile(model, mode="reduce-overhead")

    print(f"[GPU {gpu_id}] Model loaded. Starting batched generation (batch_size={BATCH_SIZE})...")

    # ── Main batched generation loop ─────────────────────────────────────
    ckpt_file = open(checkpoint_path, "a")
    completed_count = 0
    since_last_flush = 0
    pbar = tqdm(total=total_to_process, desc=f"GPU {gpu_id}", position=worker_idx, leave=True)

    try:
        while states:
            # Grab next batch of conversations needing a response
            batch = states[:BATCH_SIZE]

            # Build prompts for each conversation's current human turn
            prompts = []
            for s in batch:
                hi = s["human_turn_indices"][s["current_human_pos"]]
                human_text = s["turns"][hi]["value"]
                s["context"].append({"from": "human", "value": human_text})
                s["new_turns"].append({"from": "human", "value": human_text})
                prompts.append(format_prompt(s["context"], s["system_msg"]))

            # Tokenize with left padding
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_CONTEXT_TOKENS,
            ).to(device)

            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    stop_strings=["\nUser:"],
                    tokenizer=tokenizer,
                )

            # Decode responses and update conversation states
            input_seq_len = inputs["input_ids"].shape[1]
            finished_indices = []

            for i, s in enumerate(batch):
                generated_ids = output_ids[i, input_seq_len:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                # Clean up any residual stop string
                if "\nUser:" in response:
                    response = response[:response.index("\nUser:")].strip()

                s["new_turns"].append({"from": "gpt", "value": response})
                s["context"].append({"from": "gpt", "value": response})

                # Advance to next human turn
                s["current_human_pos"] += 1
                if s["current_human_pos"] >= len(s["human_turn_indices"]):
                    # Conversation is done — write checkpoint
                    result = {
                        "_shard_idx": s["shard_idx"],
                        "conversations": s["new_turns"],
                    }
                    ckpt_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    finished_indices.append(i)
                    completed_count += 1
                    since_last_flush += 1
                    pbar.update(1)

            # Remove finished conversations (iterate in reverse to keep indices valid)
            for i in sorted(finished_indices, reverse=True):
                states.pop(i)

            # Flush periodically
            if since_last_flush >= CHECKPOINT_EVERY:
                ckpt_file.flush()
                since_last_flush = 0

            # Periodic memory cleanup
            if completed_count % 500 == 0 and completed_count > 0:
                torch.cuda.empty_cache()

            # Re-sort remaining states: fewest remaining turns first (less padding waste)
            states.sort(key=lambda s: len(s["human_turn_indices"]) - s["current_human_pos"])

    finally:
        ckpt_file.flush()
        ckpt_file.close()
        pbar.close()

    print(f"[GPU {gpu_id}] Finished all {total_to_process} conversations.")


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
                    record = json.loads(line)
                    # Strip internal tracking field
                    record.pop("_shard_idx", None)
                    all_conversations.append(record)

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
        p = mp.Process(target=batched_worker, args=(gpu_id, shard, worker_idx))
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
