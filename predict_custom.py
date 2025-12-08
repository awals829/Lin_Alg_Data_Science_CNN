import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from CNN_Class import SimpleCNN

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
print("Device:", device)

state_dict = torch.load("models\simple_cnn_mnist.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

def center_digit(t: torch.Tensor) -> torch.Tensor:
    """
    t: [1, 28, 28] in [0,1] (white digit on black)
    returns: [1, 28, 28], digit centered
    """
    img = t[0]  # [28, 28]
    # threshold to find "ink"
    ys, xs = (img > 0.1).nonzero(as_tuple=True)
    if len(ys) == 0:
        return t  # blank image

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    digit = img[y_min:y_max+1, x_min:x_max+1]  # cropped region

    # resize cropped digit to, say, 20x20
    digit_pil = transforms.ToPILImage()(digit.unsqueeze(0))
    digit_resized = transforms.Resize((20, 20))(digit_pil)
    digit_resized_t = transforms.ToTensor()(digit_resized)[0]  # [20,20]

    # paste into center of 28x28 canvas
    canvas = torch.zeros_like(img)
    y_start = (28 - 20) // 2
    x_start = (28 - 20) // 2
    canvas[y_start:y_start+20, x_start:x_start+20] = digit_resized_t

    return canvas.unsqueeze(0)

# transform for custom images
transform = transforms.Compose([
    transforms.ToTensor(),                      # [1,28,28], [0,1]
    transforms.Lambda(center_digit),            # center & scale digit
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])

choice = "1"
drawings_original = f"customFigures/original/{choice}_28x28.png"
#drawings_svd = "customFigures/SVDConv/1_28x28.png"

print(f"Drawing Choice: {drawings_original}")
img = Image.open(drawings_original)

# Optional, but shoes the image choice
# plt.imshow(img, cmap="gray")
# plt.title("Original custom image")
# plt.axis("off")
# plt.show()

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