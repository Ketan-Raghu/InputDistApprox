"""Merge input prompts from the ShareGPT and harmful datasets into one shuffled dataset."""

import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset


SHAREGPT_PATH = Path("data/sharegpt_llama3.1_8b/sharegpt_llama3.1_8b.jsonl")
HARMFUL_DATASET = "LLM-LAT/harmful-dataset"
OUTPUT_PATH = Path("data/merged_prompts.jsonl")


def extract_sharegpt_prompts(path: Path) -> list[dict]:
    """Extract the first human turn from each ShareGPT conversation."""
    prompts = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            for turn in entry["conversations"]:
                if turn["from"] == "human":
                    prompts.append({
                        "prompt": turn["value"],
                        "source": "sharegpt",
                    })
                    break  # only take the first human turn per conversation
    return prompts


def extract_harmful_prompts() -> list[dict]:
    """Extract prompts from the LLM-LAT/harmful-dataset."""
    ds = load_dataset(HARMFUL_DATASET, split="train")
    prompts = []
    for row in ds:
        prompts.append({
            "prompt": row["prompt"],
            "source": "harmful",
        })
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Merge ShareGPT and harmful dataset prompts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output JSONL path")
    parser.add_argument("--sharegpt-path", type=str, default=str(SHAREGPT_PATH), help="Path to ShareGPT JSONL")
    args = parser.parse_args()

    print(f"Loading ShareGPT prompts from {args.sharegpt_path} ...")
    sharegpt_prompts = extract_sharegpt_prompts(Path(args.sharegpt_path))
    print(f"  -> {len(sharegpt_prompts)} prompts")

    print(f"Loading harmful prompts from HuggingFace ({HARMFUL_DATASET}) ...")
    harmful_prompts = extract_harmful_prompts()
    print(f"  -> {len(harmful_prompts)} prompts")

    merged = sharegpt_prompts + harmful_prompts
    random.seed(args.seed)
    random.shuffle(merged)
    print(f"Merged & shuffled: {len(merged)} total prompts")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for entry in merged:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
