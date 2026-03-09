"""
Row 1 - Mean - Std
Row 2 - Mean
Row 3 - Mean + Std
Row 4 - Std
"""

import sys
import os

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "LearnedInverse"))
from reverse_distill import InverseNN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNED_DIR = os.path.join(SCRIPT_DIR, "..", "LearnedInverse")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load finetuned inverse model
    model = InverseNN().to(device)
    model.load_state_dict(
        torch.load(os.path.join(LEARNED_DIR, "inverse_ffnn_finetuned_normal.pt"),
                    map_location=device, weights_only=True)
    )
    model.eval()

    # Load logit datasets
    train_data = torch.load(os.path.join(LEARNED_DIR, "mnist_logits_train.pt"), weights_only=True)
    test_data = torch.load(os.path.join(LEARNED_DIR, "mnist_logits_test.pt"), weights_only=True)
    all_logits = torch.cat([train_data["logits"], test_data["logits"]])

    # Generate images from logits
    with torch.no_grad():
        all_generated = model(all_logits.to(device)).cpu()  # (N, 784)

    # Group by predicted digit (argmax of logits)
    predicted_digits = all_logits.argmax(dim=1)
    digit_images = {i: [] for i in range(10)}
    for img, digit in zip(all_generated, predicted_digits):
        digit_images[digit.item()].append(img.view(28, 28))

    # Compute mean and std per digit
    means = {}
    stds = {}
    for digit in range(10):
        stack = torch.stack(digit_images[digit])
        means[digit] = stack.mean(dim=0)
        stds[digit] = stack.std(dim=0)

    # Create 4x10 grid: mean-std, mean, mean+std, std
    fig, axes = plt.subplots(4, 10, figsize=(15, 6.5))
    row_labels = ["mean - std", "mean", "mean + std", "std"]

    for digit in range(10):
        images = [
            (means[digit] - stds[digit]).clamp(0, 1),
            means[digit].clamp(0, 1),
            (means[digit] + stds[digit]).clamp(0, 1),
            stds[digit] / stds[digit].max(),
        ]
        for row, img in enumerate(images):
            axes[row, digit].imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row, digit].axis("off")
            if digit == 0:
                axes[row, digit].set_ylabel(row_labels[row], fontsize=10, rotation=90, labelpad=40)
                axes[row, digit].yaxis.set_visible(True)
                axes[row, digit].tick_params(left=False, labelleft=False)
        axes[0, digit].set_title(str(digit), fontsize=12)

    fig.suptitle("Finetuned (Normal) Inverse FFNN Per-Digit Distribution (mean +/- std)", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "inverse_finetuned_normal_per_digit_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
