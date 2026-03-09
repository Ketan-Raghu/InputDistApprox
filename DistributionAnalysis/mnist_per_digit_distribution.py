"""
Row 1 - Mean - Std
Row 2 - Mean
Row 3 - Mean + Std
Row 4 - Std
"""

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def main():
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())

    # Group images by digit label
    digit_images = {i: [] for i in range(10)}
    for img, label in dataset:
        digit_images[label].append(img)

    # Compute mean and std per digit
    means = {}
    stds = {}
    for digit in range(10):
        stack = torch.stack(digit_images[digit])  # (N, 1, 28, 28)
        means[digit] = stack.mean(dim=0).squeeze(0)  # (28, 28)
        stds[digit] = stack.std(dim=0).squeeze(0)

    # Create 4x10 grid: mean-std, mean, mean+std, std
    fig, axes = plt.subplots(4, 10, figsize=(15, 6.5))
    row_labels = ["mean - std", "mean", "mean + std", "std"]

    for digit in range(10):
        images = [
            (means[digit] - stds[digit]).clamp(0, 1),
            means[digit].clamp(0, 1),
            (means[digit] + stds[digit]).clamp(0, 1),
            stds[digit] / stds[digit].max(),  # normalize std to [0,1] for visibility
        ]
        for row, img in enumerate(images):
            axes[row, digit].imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row, digit].axis("off")
            if digit == 0:
                axes[row, digit].set_ylabel(row_labels[row], fontsize=10, rotation=90, labelpad=40)
                axes[row, digit].yaxis.set_visible(True)
                axes[row, digit].tick_params(left=False, labelleft=False)
        axes[0, digit].set_title(str(digit), fontsize=12)

    fig.suptitle("MNIST Per-Digit Distribution (mean +/- std)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("DistributionAnalysis/mnist_per_digit_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to DistributionAnalysis/mnist_per_digit_distribution.png")

if __name__ == "__main__":
    main()
