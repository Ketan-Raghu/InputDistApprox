import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(os.path.join(parent_dir, "data"), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(os.path.join(parent_dir, "data"), train=False, transform=transform)

    # Build (one_hot_label, flat_image) datasets
    def make_tensor_dataset(dataset):
        images = dataset.data.float().view(-1, 784)
        # Apply same normalization as transform: (x/255 - 0.1307) / 0.3081
        images = (images / 255.0 - 0.1307) / 0.3081
        labels_onehot = torch.zeros(len(dataset), 10)
        labels_onehot.scatter_(1, dataset.targets.unsqueeze(1), 1.0)
        return TensorDataset(labels_onehot, images), dataset.targets

    train_dataset, _ = make_tensor_dataset(train_set)
    test_dataset, test_labels = make_tensor_dataset(test_set)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)

    # Train
    model = InverseNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epochs = 50

    print(f"\nTraining InverseNN (label -> image) for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for onehot, images in train_loader:
            onehot, images = onehot.to(device), images.to(device)
            pred = model(onehot)
            loss = criterion(pred, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * onehot.size(0)
            train_n += onehot.size(0)

        model.eval()
        test_loss_sum, test_n = 0.0, 0
        with torch.no_grad():
            for onehot, images in test_loader:
                onehot, images = onehot.to(device), images.to(device)
                pred = model(onehot)
                loss = criterion(pred, images)
                test_loss_sum += loss.item() * onehot.size(0)
                test_n += onehot.size(0)

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"Train MSE: {train_loss_sum / train_n:.6f}  "
            f"Test MSE: {test_loss_sum / test_n:.6f}"
        )

    model_path = os.path.join(script_dir, "inverse_labels_ffnn.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to {model_path}")

    # Evaluate and visualize
    model.eval()
    with torch.no_grad():
        test_onehot = test_dataset.tensors[0].to(device)
        test_images = test_dataset.tensors[1].to(device)
        preds = model(test_onehot)
        per_sample_mse = ((preds - test_images) ** 2).mean(dim=1)
        print(f"\nTest set reconstruction — Mean MSE: {per_sample_mse.mean().item():.6f}, "
              f"Max MSE: {per_sample_mse.max().item():.6f}")

    n_samples = 8
    fig, axes = plt.subplots(2, n_samples, figsize=(16, 4))
    for i in range(n_samples):
        orig = test_images[i].cpu().view(28, 28)
        recon = preds[i].cpu().view(28, 28)

        axes[0, i].imshow(orig, cmap="gray")
        axes[0, i].set_title(f"Original ({test_labels[i].item()})")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon, cmap="gray")
        axes[1, i].set_title(f"Generated")
        axes[1, i].axis("off")

    fig.suptitle("Original vs Label-to-Image Generated", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(script_dir, "reverse_labels_samples.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
