import torch
from torchvision import transforms


class Transformer:
    def __init__(self, k = 1, threshold = None) -> None:
        self.k = k
        self.threshold = threshold

    def __call__(self, img):
        x = transforms.functional.to_tensor(img)
        X = x[0]
        
        U, D, Vt = torch.linalg.svd(X, full_matrices=False)
        
        k = min(self.k, D.shape[0])
        Uk = U[:, :k]
        Dk = D[:k]
        Vtk = Vt[:k,:]
        
        x_approx = (Uk * Dk) @ Vtk
        
        mask = x_approx.abs() < self.threshold
                
        x_approx = x_approx.masked_fill(mask, 0.0)
        
        x_approx = x_approx.clamp(0.0, 1.0)
        
        return x_approx.unsqueeze(0)
        
    def threshold_clip(self):
        pass
    
