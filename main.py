"""
Importing Libraries and Defining Transformations for MNIST Dataset
"""
import os
from pathlib import Path
import torch, torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

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


test_dataset = torchvision.datasets.MNIST(root="./data", 
                                          train=False, 
                                          download=True, 
                                          transform=None)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

img, label = train_dataset[0]
print(train_dataset[0])
print(type(train_dataset))
print(len(train_dataset))
print(type(label), label)
print(type(img), img.shape)
# print(img)
print(train_dataset.data[0])
print(train_dataset.targets[0].item())

# Visualize raw vs transformed (after undoing normalization)
plt.subplot(1, 2, 1)
plt.imshow(train_dataset.data[0].numpy(), cmap="gray")
plt.title(f"Raw (label {train_dataset.targets[0].item()})")
plt.axis("off")

# Undo normalization just for plotting
img_unnorm = img * MNIST_STD + MNIST_MEAN

plt.subplot(1, 2, 2)
plt.imshow(img_unnorm.squeeze().numpy(), cmap="gray")
plt.title("Transformed (unnormalized view)")
plt.axis("off")

plt.tight_layout()
plt.show()