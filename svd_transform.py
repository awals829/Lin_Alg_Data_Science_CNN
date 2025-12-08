# svd_transform.py
"""
Custom SVD-based transform for MNIST-like images.

Usage:
    from svd_transform import SVDThresholdTransform

    transform = transforms.Compose([
        transforms.ToTensor(),
        SVDThresholdTransform(k=10, value_threshold=0.05),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
"""

import torch


class SVDThresholdTransform:
    """
    Apply SVD to a single-channel image, reconstruct with top-k components,
    then zero out small pixel values in the reconstructed image.

    Assumes input tensor shape: [1, H, W] with values in [0, 1].
    Returns tensor of same shape [1, H, W], still in [0, 1].
    """

    def __init__(self, k: int = 10, value_threshold: float = 0.05):
        """
        Args:
            k: number of singular values/components to keep.
            value_threshold: pixel values with |value| < this become 0.0
        """
        self.k = k
        self.value_threshold = value_threshold

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() != 3 or img.size(0) != 1:
            raise ValueError(
                f"SVDThresholdTransform expects [1, H, W] tensor, got {tuple(img.shape)}"
            )

        x2d = img[0]  # [H, W]

        # SVD
        U, S, Vh = torch.linalg.svd(x2d, full_matrices=False)

        # Keep top-k components
        k = min(self.k, S.size(0))
        U_k = U[:, :k]          # [H, k]
        S_k = S[:k]             # [k]
        Vh_k = Vh[:k, :]        # [k, W]

        # Reconstruct approximation: U_k * diag(S_k) @ Vh_k
        x_approx = (U_k * S_k) @ Vh_k   # [H, W]

        # Threshold pixel values
        mask = x_approx.abs() < self.value_threshold
        x_approx = x_approx.masked_fill(mask, 0.0)

        # Clamp back to [0, 1]
        x_approx = x_approx.clamp(0.0, 1.0)

        return x_approx.unsqueeze(0)  # [1, H, W]
