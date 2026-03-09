"""Compute per-class logit distribution stats from real MNIST data.

Loads train+test logit vectors, groups by predicted class (argmax),
computes per-class mean and Cholesky factor of covariance, and saves
to logit_distribution.pt for use by finetune_loop_normal.py.
"""

import os
import torch


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, "logit_distribution.pt")

    if os.path.exists(cache_path):
        stats = torch.load(cache_path, weights_only=True)
        if "class_means" in stats:
            print("Loaded cached per-class logit distribution stats:")
            for c in range(10):
                print(f"  class {c}: mean={stats['class_means'][c]}")
        else:
            print("Loaded cached stats (old global format) — re-run with --force to upgrade")
        return stats

    train_data = torch.load(os.path.join(script_dir, "mnist_logits_train.pt"), weights_only=True)
    test_data = torch.load(os.path.join(script_dir, "mnist_logits_test.pt"), weights_only=True)

    all_logits = torch.cat([train_data["logits"], test_data["logits"]], dim=0)
    print(f"Computing per-class stats from {all_logits.shape[0]} logit vectors")

    labels = all_logits.argmax(dim=1)

    # Per-class stats
    class_means = []  # [10, 10]
    class_L = []      # [10, 10, 10] — Cholesky factors
    class_counts = []

    for c in range(10):
        mask = labels == c
        class_logits = all_logits[mask]
        class_counts.append(class_logits.shape[0])
        mean_c = class_logits.mean(dim=0)
        cov_c = torch.cov(class_logits.T)
        # Add small ridge for numerical stability
        cov_c += 1e-4 * torch.eye(10)
        L_c = torch.linalg.cholesky(cov_c)
        class_means.append(mean_c)
        class_L.append(L_c)
        print(f"  class {c}: n={class_counts[-1]}, mean={mean_c}")

    # Also keep global stats for backward compat
    mean = all_logits.mean(dim=0)
    std = all_logits.std(dim=0)
    cov = torch.cov(all_logits.T)

    stats = {
        "mean": mean,
        "std": std,
        "cov": cov,
        "class_means": torch.stack(class_means),   # [10, 10]
        "class_L": torch.stack(class_L),            # [10, 10, 10]
    }
    torch.save(stats, cache_path)

    print(f"Saved to {cache_path}")
    return stats


if __name__ == "__main__":
    main()
