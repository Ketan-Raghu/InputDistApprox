"""Generate mnist_logits_train.pt and mnist_logits_test.pt if they don't already exist.

Each file is a dict with keys "logits" (N, 10) and "images" (N, 784),
produced by running the trained classifier (mnist_ffnn.pt) on MNIST.
"""

import sys
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mnist_ffnn import FeedForwardNN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

TRAIN_PATH = os.path.join(SCRIPT_DIR, "mnist_logits_train.pt")
TEST_PATH = os.path.join(SCRIPT_DIR, "mnist_logits_test.pt")


def generate_logit_dataset(classifier, loader, device):
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
    if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
        train_data = torch.load(TRAIN_PATH, weights_only=True)
        test_data = torch.load(TEST_PATH, weights_only=True)
        print(f"Logit datasets already exist:")
        print(f"  {TRAIN_PATH}  ({train_data['logits'].shape[0]} samples)")
        print(f"  {TEST_PATH}   ({test_data['logits'].shape[0]} samples)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Generating logit datasets from trained classifier...")

    classifier = FeedForwardNN().to(device)
    weights_path = os.path.join(PARENT_DIR, "mnist_ffnn.pt")
    classifier.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(os.path.join(PARENT_DIR, "data"), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(os.path.join(PARENT_DIR, "data"), train=False, transform=transform)

    train_logits, train_images = generate_logit_dataset(
        classifier, DataLoader(train_set, batch_size=256), device
    )
    test_logits, test_images = generate_logit_dataset(
        classifier, DataLoader(test_set, batch_size=256), device
    )

    torch.save({"logits": train_logits, "images": train_images}, TRAIN_PATH)
    torch.save({"logits": test_logits, "images": test_images}, TEST_PATH)
    print(f"Saved {TRAIN_PATH}  ({train_logits.shape[0]} samples)")
    print(f"Saved {TEST_PATH}   ({test_logits.shape[0]} samples)")


if __name__ == "__main__":
    main()
