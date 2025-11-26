"""
Importing Libraries and Defining Transformations for MNIST Dataset
"""
import torch, torchvision
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

base_dir = Path("mnist_by_class")  # PNGs will go here

# Create directory classes for PNGs
for split in ["train", "test"]:
    for digit in range(10):
        out_dir = base_dir / split / str(digit)
        out_dir.mkdir(parents=True, exist_ok=True)

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
train_dataset = torchvision.datasets.MNIST(root="./data", 
                                           train=True, 
                                           download=True, 
                                           transform=transformer)

test_dataset = torchvision.datasets.MNIST(root="./data", 
                                          train=False, 
                                          download=True, 
                                          transform=transformer)


# A second copy of the dataset WITHOUT transforms, for saving images
raw_train_dataset = torchvision.datasets.MNIST(root="./data", 
                                               train=True, 
                                               download=False, 
                                               transform=None)

raw_test_dataset  = torchvision.datasets.MNIST(root="./data", 
                                               train=False, 
                                               download=False, 
                                               transform=None)

def export_split(dataset, data_split_name: str):
    directory = "mnist_by_class"
    split_directory = directory + data_split_name
    
    if (base_dir / data_split_name).exists():
        print(f"{directory}/{data_split_name} directory already exists.")
        return

    for digit in range(10):
        (split_directory / str(digit)).mkdir(parents=True, exist_ok=True)
    
    for index , (img, label) in enumerate(tqdm(dataset, desc=f"Saving {data_split_name}")):
        output_path = split_directory / str(label)
        filename = f"{index}.png"
        img.save(output_path / filename)
    return

export_split(raw_train_dataset, "train")
export_split(raw_test_dataset, "test")

"""
Setting Up Computation Device
This checks if a CUDA-capable GPU is available and sets the device accordingly.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)