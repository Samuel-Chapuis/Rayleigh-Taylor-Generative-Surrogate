import os, torch
from torchvision.datasets import MNIST

train_full = MNIST("data", train=True, download=True).data
test = MNIST("data", train=False, download=True).data

p = torch.randperm(len(train_full))
n_train = int(0.9 * len(train_full))

train = train_full[p[:n_train]]
val   = train_full[p[n_train:]]

os.makedirs("data/MNIST/processed", exist_ok=True)
torch.save(train, "data/MNIST/processed/training.pt")
torch.save(val,   "data/MNIST/processed/validation.pt")
torch.save(test,  "data/MNIST/processed/test.pt")