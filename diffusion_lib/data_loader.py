from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class ProcessedDataset(Dataset):
    def __init__(self, root: str, train: bool = True):
        split = "training" if train else "test"
        processed_path = Path(root) / "processed" / f"{split}.pt"
        if not processed_path.exists():
            raise FileNotFoundError(
                f"Missing processed dataset file: {processed_path}. "
                "Run data_mining.py to generate it."
            )

        self.images, self.labels = torch.load(processed_path)

        if self.images.ndim == 3:
            self.images = self.images.unsqueeze(1)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int):
        img = self.images[index].float().div(255.0)
        img = (img - 0.5) * 2
        label = self.labels[index]
        return img, label


def data_loader(config):
    dataset = ProcessedDataset(config.store_path_dataset, train=True)
    loader = DataLoader(dataset, config.batch_size, shuffle=True)
    return loader