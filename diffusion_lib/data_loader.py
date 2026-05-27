from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST

def data_loader(config):
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)
    ])
    dataset = MNIST(config.store_path_dataset, download=True, train=True, transform=transform)
    loader = DataLoader(dataset, config.batch_size, shuffle=True)
    return loader