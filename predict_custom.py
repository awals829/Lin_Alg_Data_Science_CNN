import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from CNN_Class import SimpleCNN

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
print("Device:", device)

state_dict = torch.load("models\simple_cnn_mnist.pth")
model.load_state_dict(state_dict)
model.eval()

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

choice = "5"
drawings_original = f"customFigures/original/{choice}_28x28.png"
#drawings_svd = "customFigures/SVDConv/1_28x28.png"

print(f"Drawing Choice: {drawings_original}")
img = Image.open(drawings_original)

# Optional, but shoes the image choice
# plt.imshow(img, cmap="gray")
# plt.title("Original custom image")
# plt.axis("off")
# plt.show()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
])

# --- preprocess and run prediction ---
input_tensor = transform(img)            # [1, 28, 28]
input_tensor = input_tensor.unsqueeze(0) # add batch dim -> [1, 1, 28, 28]
input_tensor = input_tensor.to(device)

with torch.no_grad():
    logits = model(input_tensor)         # [1, 10]
    probs = torch.softmax(logits, dim=1) # convert to probabilities
    pred_class = probs.argmax(dim=1).item()
    top_probs, top_idxs = torch.topk(probs, k=3, dim=1)

print(f"Predicted digit: {pred_class}")

print("Top-3 predictions:")
for p, idx in zip(top_probs[0], top_idxs[0]):
    print(f"  {idx.item()}: {p.item():.4f}")