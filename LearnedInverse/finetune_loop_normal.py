"""
Self-distillation fine-tuning loop for the inverse generative model.

Loop:
  1. Sample random logit vectors
  2. InverseNN(random_logits) -> synthetic images
  3. FeedForwardNN(synthetic_images) -> "true" logits
  4. Fine-tune InverseNN on mixed real + synthetic data with KL regularization
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mnist_ffnn import FeedForwardNN
from reverse_distill import InverseNN


def load_logit_distribution(device):
    """Load precomputed per-class logit distribution stats."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "logit_distribution.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python logit_stats.py` first."
        )
    stats = torch.load(path, map_location=device, weights_only=True)
    if "class_means" not in stats:
        raise RuntimeError(
            "logit_distribution.pt is missing per-class stats. "
            "Delete it and re-run `python logit_stats.py`."
        )
    class_means = stats["class_means"].to(device)  # [10, 10]
    class_L = stats["class_L"].to(device)           # [10, 10, 10]
    return class_means, class_L


def sample_random_logits(n, device, class_means, class_L):
    """Sample logit vectors from per-class Gaussian mixture.

    Uniformly picks a class for each sample, then draws from that
    class's fitted Gaussian (mean + Cholesky factor).
    """
    classes = torch.randint(0, 10, (n,), device=device)
    z = torch.randn(n, 10, device=device)
    means = class_means[classes]       # [n, 10]
    Ls = class_L[classes]              # [n, 10, 10]
    return torch.bmm(z.unsqueeze(1), Ls.transpose(1, 2)).squeeze(1) + means


def generate_synthetic_dataset(inverse_model, classifier, n_samples, device, class_means, class_L, batch_size=512):
    """Generate (true_logits, synthetic_images) pairs.

    1. Sample random logits from per-class distribution
    2. Inverse model produces synthetic images
    3. Classifier re-scores those images to get 'true' logits
    """
    inverse_model.eval()
    classifier.eval()

    all_true_logits = []
    all_synth_images = []

    remaining = n_samples
    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_size, remaining)
            random_logits = sample_random_logits(bs, device, class_means, class_L)
            synth_images = inverse_model(random_logits)
            true_logits = classifier(synth_images.view(-1, 1, 28, 28))

            all_true_logits.append(true_logits.cpu())
            all_synth_images.append(synth_images.cpu())
            remaining -= bs

    return torch.cat(all_true_logits), torch.cat(all_synth_images)


def evaluate_on_real(inverse_model, test_data, device):
    """Evaluate reconstruction MSE on real MNIST test logits."""
    inverse_model.eval()
    with torch.no_grad():
        logits = test_data["logits"].to(device)
        images = test_data["images"].to(device)
        preds = inverse_model(logits)
        mse = ((preds - images) ** 2).mean().item()
    return mse


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Load classifier (frozen)
    classifier = FeedForwardNN().to(device)
    classifier.load_state_dict(
        torch.load(os.path.join(parent_dir, "mnist_ffnn.pt"), map_location=device, weights_only=True)
    )
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    # Load pre-trained inverse model
    inverse_model = InverseNN().to(device)
    inverse_model.load_state_dict(
        torch.load(os.path.join(script_dir, "inverse_ffnn.pt"), map_location=device, weights_only=True)
    )

    # Load real training data (60k pairs)
    train_data = torch.load(os.path.join(script_dir, "mnist_logits_train.pt"), weights_only=True)
    real_logits = train_data["logits"]
    real_images = train_data["images"]
    real_dataset = TensorDataset(real_logits, real_images)
    print(f"Loaded {len(real_dataset)} real training pairs")

    # Load real test data for evaluation
    test_data = torch.load(os.path.join(script_dir, "mnist_logits_test.pt"), weights_only=True)
    baseline_mse = evaluate_on_real(inverse_model, test_data, device)
    print(f"Baseline real-data test MSE: {baseline_mse:.6f}")

    # Load per-class logit distribution for data-driven sampling
    class_means, class_L = load_logit_distribution(device)
    print(f"Loaded per-class logit distribution (10 classes)")

    # Fine-tuning config
    outer_rounds = 10
    synth_per_round = 60000  # 2x real data -> 33% real / 67% synthetic
    finetune_epochs = 5
    batch_size = 64
    lr = 1e-4
    kl_weight = 1.0

    optimizer = optim.Adam(inverse_model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()

    print(f"\nStarting self-distillation: {outer_rounds} rounds, "
          f"{synth_per_round} synthetic + {len(real_dataset)} real samples/round, "
          f"{finetune_epochs} epochs/round, kl_weight={kl_weight}\n")

    for round_idx in range(1, outer_rounds + 1):
        # Generate synthetic dataset
        print(f"Round {round_idx}/{outer_rounds}: generating {synth_per_round} synthetic pairs...")
        true_logits, synth_images = generate_synthetic_dataset(
            inverse_model, classifier, synth_per_round, device, class_means, class_L
        )
        synth_dataset = TensorDataset(true_logits, synth_images)

        # Combine real + synthetic (33/67 split)
        combined_dataset = ConcatDataset([real_dataset, synth_dataset])
        combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        # Fine-tune on mixed data with KL regularization
        for epoch in range(1, finetune_epochs + 1):
            inverse_model.train()
            mse_sum, kl_sum, n = 0.0, 0.0, 0
            for logits_batch, images_batch in combined_loader:
                logits_batch = logits_batch.to(device)
                images_batch = images_batch.to(device)

                pred = inverse_model(logits_batch)

                # MSE loss
                mse_loss = mse_criterion(pred, images_batch)

                # KL regularization: push classifier(pred) toward input logits
                pred_logits = classifier(pred.view(-1, 1, 28, 28))
                kl_loss = F.kl_div(
                    F.log_softmax(pred_logits, dim=1),
                    F.softmax(logits_batch, dim=1),
                    reduction="batchmean",
                )

                loss = mse_loss + kl_weight * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = logits_batch.size(0)
                mse_sum += mse_loss.item() * bs
                kl_sum += kl_loss.item() * bs
                n += bs

            real_mse = evaluate_on_real(inverse_model, test_data, device)
            print(f"  Epoch {epoch}/{finetune_epochs}  "
                  f"MSE: {mse_sum / n:.6f}  KL: {kl_sum / n:.6f}  "
                  f"Real Test MSE: {real_mse:.6f}")

    # Save fine-tuned model
    save_path = os.path.join(script_dir, "inverse_ffnn_finetuned_normal_50.pt")
    torch.save(inverse_model.state_dict(), save_path)
    final_mse = evaluate_on_real(inverse_model, test_data, device)
    print(f"\nFinal real-data test MSE: {final_mse:.6f} (baseline was {baseline_mse:.6f})")
    print(f"Saved fine-tuned model to {save_path}")

    # Visualize
    inverse_model.eval()
    with torch.no_grad():
        test_logits = test_data["logits"][:8].to(device)
        test_images = test_data["images"][:8].to(device)
        preds = inverse_model(test_logits)

    n_samples = 8
    fig, axes = plt.subplots(2, n_samples, figsize=(16, 4))
    for i in range(n_samples):
        orig = test_images[i].cpu().view(28, 28)
        recon = preds[i].cpu().view(28, 28)

        axes[0, i].imshow(orig, cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon, cmap="gray")
        axes[1, i].set_title("Finetuned")
        axes[1, i].axis("off")

    fig.suptitle("Original vs Self-Distillation Finetuned", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(script_dir, "finetune_loop_normal_samples_50.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
