import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import checks.verify_forward
from tensor_forward import TensorModel
from differentiable_forward import DifferentiableModel

def check_same(classify1, classify2, num_imgs=100, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    cifar10_test = datasets.CIFAR10(root='/tmp/data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

    total = 0
    correct = 0
    for x,y in test_loader:
        l1 = classify1(x)
        l2 = classify2(x)
        assert l1.argmax(1).cpu().tolist() == l2.argmax(1).cpu().tolist()

if __name__ == "__main__":
    check_same(TensorModel(), DifferentiableModel(), num_imgs=1000)
    print("All tests passed.")
    
