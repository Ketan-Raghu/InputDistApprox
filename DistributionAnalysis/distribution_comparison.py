"""
Heatmaps (distribution_comparison_heatmaps.png) — 4 rows per digit:
- Row 1: |mean difference| — where the average pixel values diverge
- Row 2: |std difference| — where variance disagrees (the inverse model has much less variance, visible as bright regions along digit strokes)
- Row 3: Per-pixel KL divergence (linear scale) — mostly dominated by a few extreme pixels
- Row 4: KL divergence (log scale) — reveals the spatial structure more clearly, with edges/corners being the worst

Bar charts (distribution_comparison_summary.png) — per-digit aggregates showing:
- Digits 1, 7, 9, 3 have the highest KL divergence — the inverse model struggles most with these
- Mean differences are small across the board (~0.006), so the model gets the average right
- Std differences are much larger (~0.06), confirming the inverse model collapses diversity — it produces less varied outputs than real MNIST
"""

import sys
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "LearnedInverse"))
from reverse_distill import InverseNN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNED_DIR = os.path.join(SCRIPT_DIR, "..", "LearnedInverse")
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")

# Normalization constants used during training
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
    """KL(P || Q) for univariate Gaussians, computed element-wise."""
    eps = 1e-8
    sigma_p = sigma_p.clamp(min=eps)
    sigma_q = sigma_q.clamp(min=eps)
    return (
        torch.log(sigma_q / sigma_p)
        + (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
        - 0.5
    )


def compute_digit_stats(images_by_digit):
    """Given {digit: tensor(N,28,28)}, return {digit: (mean, std)}."""
    stats = {}
    for digit in range(10):
        stack = torch.stack(images_by_digit[digit])
        stats[digit] = (stack.mean(dim=0), stack.std(dim=0))
    return stats


def get_mnist_stats():
    """Load MNIST and compute per-digit (mean, std) in [0,1] pixel space."""
    dataset = datasets.MNIST(
        os.path.join(PARENT_DIR, "data"), train=True, download=True,
        transform=transforms.ToTensor(),
    )
    digit_images = {i: [] for i in range(10)}
    for img, label in dataset:
        digit_images[label].append(img.squeeze(0))  # (28,28)
    return compute_digit_stats(digit_images)


def get_inverse_stats(model_filename="inverse_ffnn.pt"):
    """Generate images via inverse model, un-normalize, compute per-digit stats."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InverseNN().to(device)
    model.load_state_dict(
        torch.load(os.path.join(LEARNED_DIR, model_filename),
                    map_location=device, weights_only=True)
    )
    model.eval()

    train_data = torch.load(os.path.join(LEARNED_DIR, "mnist_logits_train.pt"), weights_only=True)
    test_data = torch.load(os.path.join(LEARNED_DIR, "mnist_logits_test.pt"), weights_only=True)
    all_logits = torch.cat([train_data["logits"], test_data["logits"]])

    with torch.no_grad():
        generated = model(all_logits.to(device)).cpu()  # (N, 784)

    # Un-normalize to [0,1] pixel space to match MNIST
    generated = generated * MNIST_STD + MNIST_MEAN

    predicted_digits = all_logits.argmax(dim=1)
    digit_images = {i: [] for i in range(10)}
    for img, digit in zip(generated, predicted_digits):
        digit_images[digit.item()].append(img.view(28, 28))

    return compute_digit_stats(digit_images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="inverse_ffnn.pt",
                        help="Model weights filename in LearnedInverse/ (default: inverse_ffnn.pt)")
    args = parser.parse_args()

    model_filename = args.model
    model_label = model_filename.replace(".pt", "").replace("_", " ").title()

    print("Computing MNIST stats...")
    mnist_stats = get_mnist_stats()
    print(f"Computing inverse model stats ({model_filename})...")
    inv_stats = get_inverse_stats(model_filename)

    # --- Per-digit KL divergence, mean diff, std diff ---
    kl_maps = {}       # per-pixel KL(MNIST || Inverse)
    mean_diffs = {}    # |mu_mnist - mu_inv|
    std_diffs = {}     # |sigma_mnist - sigma_inv|
    kl_totals = {}     # sum of per-pixel KL per digit

    for digit in range(10):
        mu_m, sig_m = mnist_stats[digit]
        mu_i, sig_i = inv_stats[digit]
        kl = gaussian_kl(mu_m, sig_m, mu_i, sig_i)
        kl_maps[digit] = kl
        mean_diffs[digit] = (mu_m - mu_i).abs()
        std_diffs[digit] = (sig_m - sig_i).abs()
        kl_totals[digit] = kl.sum().item()

    # ===== Figure 1: Heatmap grid (4 rows x 10 cols) =====
    fig, axes = plt.subplots(4, 10, figsize=(15, 7))
    row_labels = ["|mean diff|", "|std diff|", "KL(MNIST||Inv)", "KL (log scale)"]

    for digit in range(10):
        md = mean_diffs[digit].numpy()
        sd = std_diffs[digit].numpy()
        kl = kl_maps[digit].numpy()
        kl_log = np.log1p(kl)

        for row, (data, cmap) in enumerate(zip(
            [md, sd, kl, kl_log],
            ["hot", "hot", "inferno", "inferno"],
        )):
            im = axes[row, digit].imshow(data, cmap=cmap)
            axes[row, digit].axis("off")
            if digit == 0:
                axes[row, digit].set_ylabel(row_labels[row], fontsize=9,
                                            rotation=90, labelpad=40)
                axes[row, digit].yaxis.set_visible(True)
                axes[row, digit].tick_params(left=False, labelleft=False)
        axes[0, digit].set_title(str(digit), fontsize=12)

    fig.suptitle(f"Distribution Comparison: MNIST vs {model_label}", fontsize=14, y=1.02)
    plt.tight_layout()
    suffix = model_filename.replace(".pt", "")
    path1 = os.path.join(SCRIPT_DIR, f"distribution_comparison_heatmaps_{suffix}.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {path1}")

    # ===== Figure 2: Summary bar chart =====
    digits = list(range(10))
    kl_vals = [kl_totals[d] for d in digits]
    mean_diff_vals = [mean_diffs[d].mean().item() for d in digits]
    std_diff_vals = [std_diffs[d].mean().item() for d in digits]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(digits, kl_vals, color="steelblue")
    axes[0].set_title("Total KL Divergence per Digit")
    axes[0].set_xlabel("Digit")
    axes[0].set_ylabel("KL(MNIST || Inverse)")
    axes[0].set_xticks(digits)

    axes[1].bar(digits, mean_diff_vals, color="coral")
    axes[1].set_title("Mean |Mean Diff| per Digit")
    axes[1].set_xlabel("Digit")
    axes[1].set_ylabel("Avg |mu_mnist - mu_inv|")
    axes[1].set_xticks(digits)

    axes[2].bar(digits, std_diff_vals, color="mediumseagreen")
    axes[2].set_title("Mean |Std Diff| per Digit")
    axes[2].set_xlabel("Digit")
    axes[2].set_ylabel("Avg |sigma_mnist - sigma_inv|")
    axes[2].set_xticks(digits)

    fig.suptitle(f"Per-Digit Distribution Divergence Summary: {model_label}", fontsize=14)
    plt.tight_layout()
    path2 = os.path.join(SCRIPT_DIR, f"distribution_comparison_summary_{suffix}.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {path2}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Digit':>5}  {'KL Total':>12}  {'Mean |Δμ|':>12}  {'Mean |Δσ|':>12}")
    print("-" * 60)
    for d in digits:
        print(f"{d:>5}  {kl_totals[d]:>12.2f}  {mean_diff_vals[d]:>12.4f}  {std_diff_vals[d]:>12.4f}")
    print("-" * 60)
    print(f"{'Avg':>5}  {np.mean(kl_vals):>12.2f}  {np.mean(mean_diff_vals):>12.4f}  {np.mean(std_diff_vals):>12.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
