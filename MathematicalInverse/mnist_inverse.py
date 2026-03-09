import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mnist_ffnn_squareNet import FeedForwardNN


def build_inverse(state_dict, device):
    """Extract weights/biases and precompute pseudoinverses."""
    L1 = state_dict["net.0.weight"].to(device)
    b1 = state_dict["net.0.bias"].to(device)
    L2 = state_dict["net.2.weight"].to(device)
    b2 = state_dict["net.2.bias"].to(device)
    L3 = state_dict["net.4.weight"].to(device)
    b3 = state_dict["net.4.bias"].to(device)

    L1_inv = torch.linalg.pinv(L1)
    L2_inv = torch.linalg.pinv(L2)
    L3_inv = torch.linalg.pinv(L3)

    return (b1, b2, b3), (L1_inv, L2_inv, L3_inv)


def inverse_pass(O, biases, inverses):
    """Generate an image from a logit vector by inverting the network.

    Inverse of:  Z1 = f(L1 @ X + b1),  Z2 = f(L2 @ Z1 + b2),  O = L3 @ Z2 + b3

    Steps:
        Z2 = pinv(L3) @ (O - b3)
        Z1 = pinv(L2) @ (f_inv(Z2) - b2)
        X  = pinv(L1) @ (f_inv(Z1) - b1)
    """
    b1, b2, b3 = biases
    L1_inv, L2_inv, L3_inv = inverses
    f_inv = nn.LeakyReLU(negative_slope=100)

    # O shape: (batch, 10) -> transpose for matmul: (10, batch)
    O_t = O.T

    Z2 = L3_inv @ (O_t - b3.unsqueeze(1))
    Z1 = L2_inv @ (f_inv(Z2) - b2.unsqueeze(1))
    X_hat = L1_inv @ (f_inv(Z1) - b1.unsqueeze(1))

    return X_hat.T  # (batch, 784)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained model
    model = FeedForwardNN().to(device)
    model.load_state_dict(torch.load("mnist_ffnn_squareNet.pt", map_location=device, weights_only=True))
    model.eval()

    # Build inverse components
    biases, inverses = build_inverse(model.state_dict(), device)

    # Load test data and run forward pass to get logit vectors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16)

    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        X = images.view(images.size(0), -1)

        # Forward pass through trained model to get logit vectors
        logits = model(images)
        print(f"Labels:      {labels.tolist()}")
        print(f"Predictions: {logits.argmax(1).tolist()}")

        # Generate images from logit vectors using the inverse pass
        X_hat = inverse_pass(logits, biases, inverses)

        # Compare generated images with originals
        mse_per_sample = ((X - X_hat) ** 2).mean(dim=1)
        print(f"\nReconstruction MSE per sample: {[f'{v:.6f}' for v in mse_per_sample.tolist()]}")
        print(f"Mean MSE: {mse_per_sample.mean().item():.6f}")
        print(f"Max MSE:  {mse_per_sample.max().item():.6f}")

        # Show a sample of original vs generated images
        n_samples = 5
        fig, axes = plt.subplots(2, n_samples, figsize=(12, 5))
        for i in range(n_samples):
            orig = X[i].cpu().view(28, 28)
            recon = X_hat[i].cpu().view(28, 28)

            axes[0, i].imshow(orig, cmap="gray")
            axes[0, i].set_title(f"Original ({labels[i].item()})")
            axes[0, i].axis("off")

            axes[1, i].imshow(recon, cmap="gray")
            axes[1, i].set_title(f"Generated ({logits[i].argmax().item()})")
            axes[1, i].axis("off")

        fig.suptitle("Original vs Inverse-Generated Images")
        plt.tight_layout()
        plt.savefig("inverse_samples.png", dpi=150)
        print("\nSaved comparison to inverse_samples.png")
        plt.show()


if __name__ == "__main__":
    main()
