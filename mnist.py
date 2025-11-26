"""
Importing Libraries and Defining Transformations for MNIST Dataset
"""
import torch, torchvision
from torchvision import transforms


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
# A copy of the dataset WITH transforms, for saving tensors
train_dataset = torchvision.datasets.MNIST(root="./data/transformed", 
                                           train=True, 
                                           download=True, 
                                           transform=transformer)

test_dataset = torchvision.datasets.MNIST(root="./data/transformed", 
                                          train=False, 
                                          download=True, 
                                          transform=transformer)


# A second copy of the dataset WITHOUT transforms, for saving images
raw_train_dataset = torchvision.datasets.MNIST(root="./data/raw", 
                                               train=True, 
                                               download=True, 
                                               transform=None)

raw_test_dataset  = torchvision.datasets.MNIST(root="./data/raw", 
                                               train=False, 
                                               download=True, 
                                               transform=None)

"""
Setting Up Computation Device
This checks if a CUDA-capable GPU is available and sets the device accordingly.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)