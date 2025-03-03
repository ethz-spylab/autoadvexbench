import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import checks.verify_forward
from tensor_forward import TensorModel
from differentiable_forward import DifferentiableModel
from checks.utils import SHAPE_TORCH

def test_gradient_nonzero(model):
    x = torch.rand(1, *SHAPE_TORCH, requires_grad=True)
    y = torch.tensor([3])  # Arbitrary target class
    
    logits = model(x)
    loss = logits.square().sum()
    loss.backward()

    assert x.grad is not None, "FAILS Gradients should not be None"
    assert torch.abs(x.grad).sum().item() != 0, "FAILS Gradients should not be zero"
        

if __name__ == "__main__":
    test_gradient_nonzero(DifferentiableModel())
    print("PASSES")
    
