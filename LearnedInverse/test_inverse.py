"""
Interactive test tool: input a logit vector and visualize outputs from
both the baseline and finetuned inverse models side by side.
"""

import sys
import os

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reverse_distill import InverseNN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_models(device):
    baseline = InverseNN().to(device)
    baseline.load_state_dict(
        torch.load(os.path.join(SCRIPT_DIR, "inverse_ffnn.pt"), map_location=device, weights_only=True)
    )
    baseline.eval()

    finetuned = InverseNN().to(device)
    finetuned.load_state_dict(
        torch.load(os.path.join(SCRIPT_DIR, "inverse_ffnn_finetuned.pt"), map_location=device, weights_only=True)
    )
    finetuned.eval()

    return baseline, finetuned


def parse_logits(raw: str) -> torch.Tensor:
    """Parse a logit vector from user input.

    Accepts either:
      - 10 comma/space-separated floats (full logit vector)
      - A single integer 0-9 (creates a one-hot-style vector with that class dominant)
    """
    raw = raw.strip()
    parts = raw.replace(",", " ").split()

    if len(parts) == 1:
        digit = int(parts[0])
        if not 0 <= digit <= 9:
            raise ValueError("Single digit must be 0-9")
        logits = torch.zeros(10)
        logits[digit] = 10.0
        return logits

    if len(parts) == 10:
        return torch.tensor([float(x) for x in parts])

    raise ValueError("Enter either a single digit (0-9) or exactly 10 float values")


def show(logits: torch.Tensor, baseline, finetuned, device):
    with torch.no_grad():
        inp = logits.unsqueeze(0).to(device)
        img_base = baseline(inp).cpu().view(28, 28)
        img_fine = finetuned(inp).cpu().view(28, 28)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img_base, cmap="gray")
    axes[0].set_title("Baseline")
    axes[0].axis("off")
    axes[1].imshow(img_fine, cmap="gray")
    axes[1].set_title("Finetuned")
    axes[1].axis("off")

    label = f"[{', '.join(f'{v:.1f}' for v in logits)}]"
    fig.suptitle(f"Input logits: {label}", fontsize=9)
    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline, finetuned = load_models(device)
    print("Loaded baseline and finetuned inverse models.")
    print("Enter a logit vector (10 floats) or a single digit (0-9).")
    print("Type 'q' to quit.\n")

    while True:
        # try:
        #     raw = input("logits> ")
        # except (EOFError, KeyboardInterrupt):
        #     print()
        #     break

        # if raw.strip().lower() in ("q", "quit", "exit"):
        #     break

        try:
            # logits = parse_logits(raw)
            logits = torch.tensor([1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.])
        except ValueError as e:
            print(f"  Error: {e}")
            continue

        print(f"  Parsed: {logits.tolist()}")
        show(logits, baseline, finetuned, device)


if __name__ == "__main__":
    main()
