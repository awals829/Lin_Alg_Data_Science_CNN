import torch
import torch.nn as nn
import torch.optim as optim

# --- Simple CNN model for MNIST ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (batch, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # -> (16, 28, 28)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # -> (32, 28, 28)
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 digits

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)            # raw scores (logits)
        return x