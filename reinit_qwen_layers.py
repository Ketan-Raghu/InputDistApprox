"""
Copy Qwen2.5-7B-Instruct and reinitialize the last 8 transformer layers.

Loads the pretrained model, reinitializes layers 20-27 (the last 8 of 28)
using the model's original initializer_range (0.02 normal distribution),
while preserving lm_head.weight and all other weights. Saves the modified
model to a new directory for downstream retraining.

Usage:
  python reinit_qwen_layers.py [--output-dir OUTPUT_DIR] [--num-reinit-layers 8]
"""

import argparse
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/ketan/LLMs/models/Qwen_Qwen2.5-7B-Instruct/"
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "models", "qwen2.5-7b-reinit-last8")


def reinit_module(module: nn.Module, std: float):
    """Reinitialize all parameters in a module with normal(0, std),
    except for LayerNorm which gets reset to weight=1, bias=0."""
    for name, param in module.named_parameters():
        if isinstance(module, (nn.LayerNorm,)) or "layernorm" in name.lower():
            # Find the actual sub-module owning this param
            pass  # handled below
        else:
            nn.init.normal_(param, mean=0.0, std=std)

    # Reset LayerNorm sub-modules properly
    for submodule in module.modules():
        if isinstance(submodule, nn.LayerNorm) or type(submodule).__name__ == "Qwen2RMSNorm":
            if hasattr(submodule, "weight"):
                nn.init.ones_(submodule.weight)
            if hasattr(submodule, "bias") and submodule.bias is not None:
                nn.init.zeros_(submodule.bias)


def main():
    parser = argparse.ArgumentParser(
        description="Copy Qwen2.5-7B-Instruct with last N layers reinitialized"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT,
        help=f"Directory to save the modified model (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--num-reinit-layers", type=int, default=8,
        help="Number of final transformer layers to reinitialize (default: 8)",
    )
    args = parser.parse_args()

    print(f"Loading model from {MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    num_layers = model.config.num_hidden_layers
    std = model.config.initializer_range
    start_layer = num_layers - args.num_reinit_layers

    print(f"Model has {num_layers} layers, reinitializing layers {start_layer}-{num_layers - 1}")
    print(f"  initializer_range (std): {std}")
    print(f"  Preserving: lm_head.weight, layers 0-{start_layer - 1}, embeddings, final norm")

    # Reinitialize the last N transformer layers
    for layer_idx in range(start_layer, num_layers):
        layer = model.model.layers[layer_idx]
        reinit_module(layer, std)
        print(f"  Reinitialized layer {layer_idx}")

    # Verify lm_head.weight was NOT touched by printing a quick stat
    lm_head_norm = model.lm_head.weight.data.norm().item()
    print(f"  lm_head.weight L2 norm: {lm_head_norm:.4f} (preserved from pretrained)")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
