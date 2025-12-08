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
transformer = transforms.Compose([
    # Data augmentation in pixel space (on PIL images)
    transforms.RandomAffine(
        degrees=15,               # small rotations
        translate=(0.1, 0.1),     # up to 10% shift in x/y
        scale=(0.9, 1.1),         # small zoom in/out
        shear=10,                 # mild shear (skew)
        fill=0                    # keep background black
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
])


"""
MNIST Datasets and Device Setup
This module sets up the MNIST datasets for training and testing,
applying necessary transformations, and determines the computation device.
"""
train_dataset = torchvision.datasets.MNIST(root="./data", 
                                           train=True, 
                                           download=True, 
                                           transform=transformer)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


"""
For the Write-Up portion we can use this.
"""
choice = 2
img, label = train_dataset[choice]
print(train_dataset[choice])
print(type(train_dataset))
print(len(train_dataset))
print(type(label), label)
print(type(img), img.shape)
# print(img)
print(train_dataset.data[choice])
print(train_dataset.targets[choice].item())

# Visualize raw vs transformed (after undoing normalization)
plt.subplot(1, 2, 1)
plt.imshow(train_dataset.data[choice].numpy(), cmap="gray")
plt.title(f"Raw (label {train_dataset.targets[choice].item()})")
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

    for batch_idx, (images, labels) in enumerate(train_dataloader):
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
                  f"Step [{batch_idx+1}/{len(train_dataloader)}], "
                  f"Loss: {avg_loss:.4f}")
            running_loss = 0.0

        # OPTIONAL: stop early to just test GPU works quickly
        # if batch_idx == 200:
        #     break

print("Finished training run.")

# --- Save model weights ---
save_dir = Path("models")
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "simple_cnn_mnist.pth"

torch.save(model.state_dict(), save_path)
print(f"Saved model state_dict to: {save_path}")