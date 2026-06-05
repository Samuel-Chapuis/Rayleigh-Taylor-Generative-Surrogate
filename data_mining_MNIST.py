import os, torch
from torchvision.datasets import MNIST

os.makedirs("data/MNIST/processed", exist_ok=True)
torch.save(MNIST("data", train=True,  download=True).data, "data/MNIST/processed/training.pt")
torch.save(MNIST("data", train=False, download=True).data, "data/MNIST/processed/test.pt")