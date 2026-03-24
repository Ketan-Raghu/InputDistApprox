"""Convert a SHARDED_STATE_DICT FSDP checkpoint to HuggingFace format.

Usage:
    python convert_sharded_checkpoint.py checkpoints/inverse_mapping/stage2_step1000_old
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.config import TrainingConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: <output_dir>/final_model)",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    model_shard_dir = ckpt_dir / "pytorch_model_fsdp_0"
    if not model_shard_dir.exists():
        print(f"Error: {model_shard_dir} not found", file=sys.stderr)
        sys.exit(1)

    config = TrainingConfig()
    output_dir = args.output_dir or str(Path(config.output_dir) / "final_model")

    print(f"Loading base model from {config.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    print(f"Loading sharded checkpoint from {model_shard_dir}...")
    state_dict = {"model": model.state_dict()}
    dcp.load(state_dict, checkpoint_id=str(model_shard_dir))
    model.load_state_dict(state_dict["model"])
    del state_dict

    print(f"Saving HuggingFace model to {output_dir}...")
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = config.pad_token
    tokenizer.pad_token_id = config.pad_token_id
    tokenizer.save_pretrained(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
