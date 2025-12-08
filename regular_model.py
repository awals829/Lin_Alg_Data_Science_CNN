<<<<<<< HEAD
"""
Importing Libraries and Defining Transformations for MNIST Dataset
"""
import os
from pathlib import Path
import torch, torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from CNN_Class import SimpleCNN
import torch.nn as nn
import torch.optim as optim
from svd_transform import SVDThresholdTransform


"""
Setting Up Computation Device
This checks if a CUDA-capable GPU is available and sets the device accordingly.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


"""
Defining Mean and Standard Deviation for MNIST Normalization
these values are standard for the MNIST dataset.
"""
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


"""
Defining Transformation Pipeline for MNIST Dataset
This pipeline converts images to tensors and normalizes them using the defined mean and standard deviation.
"""
svd_transformer = transforms.Compose([
    # Data augmentation in pixel space (on PIL images)
    transforms.RandomAffine(
        degrees=15,               # small rotations
        translate=(0.1, 0.1),     # up to 10% shift in x/y
        scale=(0.9, 1.1),         # small zoom in/out
        shear=10,                 # mild shear (skew)
        fill=0                    # keep background black
    ),
    transforms.ToTensor(),
    SVDThresholdTransform(5, 0.05),
    transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
])


"""
MNIST Datasets and Device Setup
This module sets up the MNIST datasets for training and testing,
applying necessary transformations, including the custom SVD,
and determines the computation device.
"""
svd_train_dataset = torchvision.datasets.MNIST(root="./data", 
                                           train=True, 
                                           download=True, 
                                           transform=svd_transformer)

svd_train_dataloader = torch.utils.data.DataLoader(svd_train_dataset, batch_size=64, shuffle=True)


"""
For the Write-Up portion we can use this.
"""
choice = 2
img, label = svd_train_dataset[choice]
print(svd_train_dataset[choice])
print(type(svd_train_dataset))
print(len(svd_train_dataset))
print(type(label), label)
print(type(img), img.shape)
# print(img)
print(svd_train_dataset.data[choice])
print(svd_train_dataset.targets[choice].item())

# Visualize raw vs transformed (after undoing normalization)
plt.subplot(1, 2, 1)
plt.imshow(svd_train_dataset.data[choice].numpy(), cmap="gray")
plt.title(f"Raw (label {svd_train_dataset.targets[choice].item()})")
plt.axis("off")

# Undo normalization just for plotting
img_unnorm = img * MNIST_STD + MNIST_MEAN

plt.subplot(1, 2, 2)
plt.imshow(img_unnorm.squeeze().numpy(), cmap="gray")
plt.title("Transformed (unnormalized view)")
plt.axis("off")

plt.tight_layout()
plt.show()

# --- Instantiate model, loss, optimizer ---
model = SimpleCNN().to(device)               # move model to GPU or CPU
criterion = nn.CrossEntropyLoss()            # good for classification
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 15

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(svd_train_dataloader):
        # Move data to same device as model
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)          # shape: (batch_size, 10)
        loss = criterion(outputs, labels)

        # Backward + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print every 100 batches so you see it's doing something
        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / 100
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Step [{batch_idx+1}/{len(svd_train_dataloader)}], "
                  f"Loss: {avg_loss:.4f}")
            running_loss = 0.0

        # OPTIONAL: stop early to just test GPU works quickly
        # if batch_idx == 200:
        #     break

print("Finished training run.")

# --- Save model weights ---
save_dir = Path("models")
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "svd_simple_cnn_mnist.pth"

torch.save(model.state_dict(), save_path)
print(f"Saved model state_dict to: {save_path}")
=======
# main.py
"""
Train a SimpleCNN on MNIST (with affine data augmentation) and save weights.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from CNN_Class import SimpleCNN, MNIST_MEAN, MNIST_STD, get_device


# --- Device setup ---
device = get_device()
print("Device:", device)


# --- Transform: MNIST + augmentation for robustness ---
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=15,           # small rotations
        translate=(0.1, 0.1), # up to 10% shift in x/y
        scale=(0.9, 1.1),     # small zoom
        shear=10,             # mild skew
        fill=0                # keep background black
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=(MNIST_MEAN,), std=(MNIST_STD,))
])

# --- Dataset & DataLoader ---
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)


# --- Optional: inspect one sample (raw vs transformed) ---
def inspect_sample(idx: int = 2) -> None:
    img_t, label = train_dataset[idx]
    print("Sample label:", label)
    print("Transformed tensor shape:", img_t.shape)

    # left: raw data from .data (uint8 0–255)
    plt.subplot(1, 2, 1)
    plt.imshow(train_dataset.data[idx].numpy(), cmap="gray")
    plt.title(f"Raw (label {train_dataset.targets[idx].item()})")
    plt.axis("off")

    # right: transformed (undo normalization so it looks normal)
    img_unnorm = img_t * MNIST_STD + MNIST_MEAN
    plt.subplot(1, 2, 2)
    plt.imshow(img_unnorm.squeeze().numpy(), cmap="gray")
    plt.title("Transformed (unnormalized view)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Uncomment for quick visual sanity check:
# inspect_sample(2)


# --- Model, loss, optimizer ---
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(num_epochs: int = 15) -> None:
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # (batch, 10)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Step [{batch_idx+1}/{len(train_dataloader)}], "
                    f"Loss: {avg_loss:.4f}"
                )
                running_loss = 0.0

    print("Finished training run.")

"""
Adjust model name here to save a different model
"""
def save_model(path: Path = Path("models/simple_cnn_mnist.pth")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model state_dict to: {path}")


if __name__ == "__main__":
    train(num_epochs=15)
    save_model()
>>>>>>> 30a8a64f4dfe0603060c7bc4e404e2d932665ef2
