import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Allow importing from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mnist_ffnn import FeedForwardNN


class InverseNN(nn.Module):
    """Mirror of the classifier: 10 -> 128 -> 256 -> 784"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 784),
        )

    def forward(self, x):
        return self.net(x)


def generate_logit_dataset(classifier, loader, device):
    """Run classifier on a DataLoader and return (logits, flat_images) tensors."""
    classifier.eval()
    all_logits = []
    all_images = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            flat = images.view(images.size(0), -1)
            logits = classifier(images)
            all_logits.append(logits.cpu())
            all_images.append(flat.cpu())
    return torch.cat(all_logits), torch.cat(all_images)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # --- Step 2: Generate logit datasets ---
    train_logits_path = os.path.join(script_dir, "mnist_logits_train.pt")
    test_logits_path = os.path.join(script_dir, "mnist_logits_test.pt")

    if os.path.exists(train_logits_path) and os.path.exists(test_logits_path):
        print("Loading cached logit datasets...")
        train_data = torch.load(train_logits_path, weights_only=True)
        test_data = torch.load(test_logits_path, weights_only=True)
    else:
        print("Generating logit datasets from trained classifier...")
        classifier = FeedForwardNN().to(device)
        weights_path = os.path.join(parent_dir, "mnist_ffnn.pt")
        classifier.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_set = datasets.MNIST(os.path.join(parent_dir, "data"), train=True, download=True, transform=transform)
        test_set = datasets.MNIST(os.path.join(parent_dir, "data"), train=False, transform=transform)

        train_logits, train_images = generate_logit_dataset(
            classifier, DataLoader(train_set, batch_size=256), device
        )
        test_logits, test_images = generate_logit_dataset(
            classifier, DataLoader(test_set, batch_size=256), device
        )

        train_data = {"logits": train_logits, "images": train_images}
        test_data = {"logits": test_logits, "images": test_images}

        torch.save(train_data, train_logits_path)
        torch.save(test_data, test_logits_path)
        print(f"Saved logit datasets ({train_logits.shape[0]} train, {test_logits.shape[0]} test)")

    train_dataset = TensorDataset(train_data["logits"], train_data["images"])
    test_dataset = TensorDataset(test_data["logits"], test_data["images"])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)

    # --- Step 3: Train the inverse network ---
    model = InverseNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    epochs = 1000

    print(f"\nTraining InverseNN for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for logits, images in train_loader:
            logits, images = logits.to(device), images.to(device)
            pred = model(logits)
            loss = criterion(pred, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * logits.size(0)
            train_n += logits.size(0)

        model.eval()
        test_loss_sum, test_n = 0.0, 0
        with torch.no_grad():
            for logits, images in test_loader:
                logits, images = logits.to(device), images.to(device)
                pred = model(logits)
                loss = criterion(pred, images)
                test_loss_sum += loss.item() * logits.size(0)
                test_n += logits.size(0)

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"Train MSE: {train_loss_sum / train_n:.6f}  "
            f"Test MSE: {test_loss_sum / test_n:.6f}"
        )

    model_path = os.path.join(script_dir, "inverse_ffnn.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved inverse model to {model_path}")

    # --- Step 4: Evaluate and visualize ---
    model.eval()
    with torch.no_grad():
        test_logits = test_data["logits"].to(device)
        test_images = test_data["images"].to(device)
        preds = model(test_logits)
        per_sample_mse = ((preds - test_images) ** 2).mean(dim=1)
        print(f"\nTest set reconstruction — Mean MSE: {per_sample_mse.mean().item():.6f}, "
              f"Max MSE: {per_sample_mse.max().item():.6f}")

    n_samples = 8
    fig, axes = plt.subplots(2, n_samples, figsize=(16, 4))
    for i in range(n_samples):
        orig = test_images[i].cpu().view(28, 28)
        recon = preds[i].cpu().view(28, 28)

        axes[0, i].imshow(orig, cmap="gray")
        axes[0, i].set_title(f"Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon, cmap="gray")
        axes[1, i].set_title(f"Generated")
        axes[1, i].axis("off")

    fig.suptitle("Original vs Reverse-Distilled Images", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(script_dir, "reverse_distill_samples.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
