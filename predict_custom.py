# predict_custom.py
"""
Predict the digit in a custom drawn 28x28 PNG using a trained SimpleCNN.

Also provides utility functions:
- load_model()
- center_digit()
- predict_image_path()
"""

from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

from CNN_Class import SimpleCNN, MNIST_MEAN, MNIST_STD, get_device


device = get_device()
print("Device:", device)


def load_model(
    weights_path: Path = Path("models/simple_cnn_mnist.pth"),
) -> nn.Module:
    """Load SimpleCNN with trained weights, moved to the correct device."""
    model = SimpleCNN().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from: {weights_path}")
    return model


def center_digit(t: torch.Tensor) -> torch.Tensor:
    """
    Center a single-channel [1, 28, 28] digit on a 28x28 canvas.

    - Finds non-background pixels (threshold > 0.1).
    - Crops to the tight bounding box.
    - Resizes the cropped digit to 20x20.
    - Pastes into the center of a 28x28 zero canvas.

    Returns:
        torch.Tensor: centered tensor of shape [1, 28, 28].
    """
    img = t[0]  # [28, 28]

    ys, xs = (img > 0.1).nonzero(as_tuple=True)
    if len(ys) == 0:
        return t  # blank image, nothing to center

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    digit = img[y_min:y_max + 1, x_min:x_max + 1]  # cropped region

    # Resize cropped digit to ~20x20
    digit_pil = transforms.ToPILImage()(digit.unsqueeze(0))   # [1,H,W] -> PIL
    digit_resized = transforms.Resize((20, 20))(digit_pil)
    digit_resized_t = transforms.ToTensor()(digit_resized)[0]  # [20, 20]

    # Paste into center of 28x28 canvas
    canvas = torch.zeros_like(img)
    y_start = (28 - 20) // 2
    x_start = (28 - 20) // 2
    canvas[y_start:y_start + 20, x_start:x_start + 20] = digit_resized_t

    return canvas.unsqueeze(0)


# Transform for custom images saved by window.py
custom_transform = transforms.Compose([
    transforms.ToTensor(),                      # [1, 28, 28], in [0,1]
    transforms.Lambda(center_digit),            # center & scale digit
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])


def predict_image_path(
    img_path: Path,
    model: nn.Module,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    Predict the digit in a custom image.

    Args:
        img_path: path to 28x28 PNG (white digit on black).
        model: loaded SimpleCNN in eval mode.

    Returns:
        (predicted_class, top_probs, top_idxs)
    """
    img = Image.open(img_path)

    # preprocess
    input_tensor = custom_transform(img)        # [1, 28, 28]
    input_tensor = input_tensor.unsqueeze(0)    # [1, 1, 28, 28]
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)            # [1, 10]
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        top_probs, top_idxs = torch.topk(probs, k=3, dim=1)

    return pred_class, top_probs[0].cpu(), top_idxs[0].cpu()


if __name__ == "__main__":
    # Simple CLI: choose a file like "customFigures/original/4_28x28.png"
    choice = "4"
    img_path = Path(f"customFigures/original/{choice}_28x28.png")
    print(f"Predicting for: {img_path}")

    model = load_model()
    pred_class, top_probs, top_idxs = predict_image_path(img_path, model)

    print(f"Predicted digit: {pred_class}")
    print("Top-3 predictions:")
    for p, idx in zip(top_probs, top_idxs):
        print(f"  {idx.item()}: {p.item():.4f}")
