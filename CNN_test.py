from CNN_Class import SimpleCNN
import torch, torchvision
from torchvision import datasets, transforms

device= "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
print(device)

state_dict = torch.load("models\simple_cnn_mnist.pth")
model.load_state_dict(state_dict)
model.eval()

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
])

test_dataset = torchvision.datasets.MNIST(
                                        root="./data", 
                                        train=False, 
                                        download=True, 
                                        transform=test_transform
                                        )

test_loader = torch.utils.data.DataLoader(
                                        test_dataset,
                                        batch_size=128,
                                        shuffle=False
                                        )

correct_answers = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)               # (batch, 10)
        _, predicted = torch.max(outputs, 1)  # pick class with highest logit

        total += labels.size(0)
        correct_answers += (predicted == labels).sum().item()

accuracy = 100.0 * correct_answers / total
print(f"Test Accuracy: {accuracy:.2f}%")