# CNN_Class.py
import torch
import torch.nn as nn

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081  # shared constants


class SimpleCNN(nn.Module):
    """
    A small CNN for MNIST:
    - Input: (N, 1, 28, 28)
    - 2 conv layers + max pool
    - 2 fully connected layers to 10 logits (digits 0-9)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # -> (16, 28, 28)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # -> (32, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)                            # -> (32, 14, 14)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
