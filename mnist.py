import torch, torchvision
from torchvision import transforms

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transformer)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transformer)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)