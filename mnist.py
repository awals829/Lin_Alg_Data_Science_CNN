"""
Importing Libraries and Defining Transformations for MNIST Dataset
"""
import os
import torch, torchvision
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from Transformer import Transformer

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

transformer_svd = transforms.Compose([
    Transformer(k=8, threshold=0.5),
    transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
])

"""
MNIST Datasets and Device Setup
This module sets up the MNIST datasets for training and testing,
applying necessary transformations, and determines the computation device.


What appears in the data folder after download:

- t10k-images-idx3-ubyte
    -- 't10k': test 10k (the 10,000 test images).

- train-images-idx3-ubyte
    -- 'train': the training split (60,000 images)
    -- 'images': the pixel values
    -- 'idx3': the 3-dimension array (num_images, rows, cols)
    -- 'ubyte': the data type (unsigned byte) uint8 values between 0-255

- train-labels-idx1-ubyte
    -- 'idx1': 1-dimensional array: just the labels (num_images,)
    -- Each entry is a single byte 0-9 (the digit class).
"""
train_dataset = torchvision.datasets.MNIST(root="./data", 
                                           train=True, 
                                           download=True,
                                           transform=None)

test_dataset = torchvision.datasets.MNIST(root="./data", 
                                          train=False, 
                                          download=True, 
                                          transform=None)

# Pick an index to inspect
idx = 0
img, label = train_dataset[idx]
print("Label:", label)

# Apply both transforms
img_base = transformer(img)   # (1, 28, 28)
img_svd  = transformer_svd(img)    # (1, 28, 28)

# 4. Visualize side-by-side
fig, axs = plt.subplots(1, 2, figsize=(6, 3))

axs[0].imshow(img_base.squeeze().numpy(), cmap="gray")
axs[0].set_title("Original (ToTensor)")
axs[0].axis("off")

axs[1].imshow(img_svd.squeeze().numpy(), cmap="gray")
axs[1].set_title("SVD + pixel threshold")
axs[1].axis("off")

plt.tight_layout()
plt.show()

# 5. "Size" comparison: num elements, non-zeros, memory
def tensor_stats(t):
    numel = t.numel()
    nonzero = (t != 0).sum().item()
    bytes_ = numel * t.element_size()
    return numel, nonzero, bytes_

for name, t in [("Original", img_base), ("SVD+thresh", img_svd)]:
    numel, nonzero, bytes_ = tensor_stats(t)
    density = nonzero / numel
    print(f"{name}:")
    print(f"  shape   = {tuple(t.shape)}")
    print(f"  numel   = {numel}")
    print(f"  nonzero = {nonzero}")
    print(f"  density = {density:.3f}")
    print(f"  ~bytes  = {bytes_}")
    
N = 1
for idx in range(N):
    img, label = train_dataset[idx]  # img is PIL.Image

    # Original tensor in [0,1], shape (1, 28, 28)
    img_orig = transformer(img)

    # SVD + threshold version, also (1, 28, 28) in [0,1]
    img_svd = transformer_svd(img)

    # Build filenames
    orig_path = os.path.join("./drawings", f"mnist_{idx:04d}_label_{label}_orig.png")
    svd_path  = os.path.join("./drawings", f"mnist_{idx:04d}_label_{label}_svd.png")

    # Save images
    save_image(img_orig, orig_path)
    save_image(img_svd, svd_path)

    print(f"Saved {orig_path} and {svd_path}")