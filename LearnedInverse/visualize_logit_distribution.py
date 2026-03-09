"""Visualize the logit vector distribution: marginals, correlation matrix, and real vs sampled comparison."""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load real logits
    train_data = torch.load(os.path.join(script_dir, "mnist_logits_train.pt"), weights_only=True)
    test_data = torch.load(os.path.join(script_dir, "mnist_logits_test.pt"), weights_only=True)
    real_logits = torch.cat([train_data["logits"], test_data["logits"]], dim=0).numpy()

    # Load stats and generate correlated samples
    stats = torch.load(os.path.join(script_dir, "logit_distribution.pt"), weights_only=True)
    mean = stats["mean"]
    cov = stats["cov"]
    L = torch.linalg.cholesky(cov)
    z = torch.randn(len(real_logits), 10)
    sampled_logits = (z @ L.T + mean).numpy()

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Logit Vector Distribution Analysis", fontsize=16)

    # --- 1. Correlation matrix ---
    ax1 = fig.add_subplot(2, 3, 1)
    corr = np.corrcoef(real_logits.T)
    im = ax1.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_title("Correlation Matrix (real)")
    ax1.set_xlabel("Logit index")
    ax1.set_ylabel("Logit index")
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # --- 2. Covariance matrix ---
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(cov.numpy(), cmap="RdBu_r")
    ax2.set_title("Covariance Matrix")
    ax2.set_xlabel("Logit index")
    ax2.set_ylabel("Logit index")
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(10))
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # --- 3. Marginal distributions (real vs sampled) ---
    ax3 = fig.add_subplot(2, 3, 3)
    positions = np.arange(10)
    bp_real = ax3.boxplot(real_logits, positions=positions - 0.15, widths=0.25,
                          patch_artist=True, showfliers=False)
    bp_samp = ax3.boxplot(sampled_logits, positions=positions + 0.15, widths=0.25,
                          patch_artist=True, showfliers=False)
    for patch in bp_real["boxes"]:
        patch.set_facecolor("#4C72B0")
    for patch in bp_samp["boxes"]:
        patch.set_facecolor("#DD8452")
    ax3.set_title("Marginals: Real vs Sampled")
    ax3.set_xlabel("Logit index")
    ax3.set_ylabel("Value")
    ax3.legend([bp_real["boxes"][0], bp_samp["boxes"][0]], ["Real", "Sampled"], loc="upper right")
    ax3.set_xticks(range(10))

    # --- 4-5. Pairwise scatter for two interesting pairs ---
    # Pick the most positively and most negatively correlated pairs
    corr_triu = np.triu(corr, k=1)
    most_pos = np.unravel_index(np.argmax(corr_triu), corr_triu.shape)
    corr_triu_neg = corr_triu.copy()
    corr_triu_neg[corr_triu_neg == 0] = 1
    most_neg = np.unravel_index(np.argmin(corr_triu_neg), corr_triu_neg.shape)

    for idx, (i, j) in enumerate([most_neg, most_pos]):
        ax = fig.add_subplot(2, 3, 4 + idx)
        ax.scatter(real_logits[:2000, i], real_logits[:2000, j], alpha=0.15, s=4, label="Real")
        ax.scatter(sampled_logits[:2000, i], sampled_logits[:2000, j], alpha=0.15, s=4, label="Sampled")
        r = corr[i, j]
        ax.set_title(f"Logit {i} vs {j} (r={r:.2f})")
        ax.set_xlabel(f"Logit {i}")
        ax.set_ylabel(f"Logit {j}")
        ax.legend(markerscale=4)

    # --- 6. Mean and std comparison ---
    ax6 = fig.add_subplot(2, 3, 6)
    x = np.arange(10)
    real_mean = real_logits.mean(axis=0)
    real_std = real_logits.std(axis=0)
    samp_mean = sampled_logits.mean(axis=0)
    samp_std = sampled_logits.std(axis=0)
    ax6.errorbar(x - 0.1, real_mean, yerr=real_std, fmt="o", capsize=3, label="Real")
    ax6.errorbar(x + 0.1, samp_mean, yerr=samp_std, fmt="s", capsize=3, label="Sampled")
    ax6.set_title("Mean +/- Std per Logit")
    ax6.set_xlabel("Logit index")
    ax6.set_ylabel("Value")
    ax6.set_xticks(range(10))
    ax6.legend()

    plt.tight_layout()
    save_path = os.path.join(script_dir, "logit_distribution_viz.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
