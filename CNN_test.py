# CNN_test.py
"""
Load a trained SimpleCNN and evaluate accuracy on the MNIST test set.
"""

from pathlib import Path

import torch
import torchvision
from torchvision import transforms

from CNN_Class import SimpleCNN, MNIST_MEAN, MNIST_STD, get_device

device = get_device()
print("Device:", device)

# --- Load model ---
model = SimpleCNN().to(device)
state_path = Path("models/k2_svd_simple_cnn_mnist.pth")
state_dict = torch.load(state_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded model from: {state_path}")

# --- Test dataset & loader (no augmentation) ---
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(MNIST_MEAN,), std=(MNIST_STD,)),
])

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=test_transform,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False,
)


# --- Evaluate accuracy ---
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)               # (batch, 10)
        _, predicted = torch.max(outputs, 1)  # argmax over classes

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
