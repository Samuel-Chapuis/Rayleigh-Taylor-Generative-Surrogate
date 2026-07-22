from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class ProcessedDataset(Dataset):
    """
    Jeu de données PyTorch basé sur des fichiers déjà prétraités.

    Structure attendue::

        root/
            processed/
                training.pt
                validation.pt
                test.pt

    Les images sont normalisées dans l'intervalle [-1, 1].
    """

    VALID_SPLITS = {"training", "validation", "test"}

    def __init__(
        self,
        root: str | Path,
        train: bool | None = None,
        split: str | None = None,
    ) -> None:
        split = self._resolve_split(train=train, split=split)
        processed_path = self._build_processed_path(root=root, split=split)

        self.split = split
        self.processed_path = processed_path
        self.images, self.labels = self._load_processed_file(processed_path)

        if self.images.ndim == 3:
            self.images = self.images.unsqueeze(1)

    @classmethod
    def _resolve_split(
        cls,
        train: bool | None,
        split: str | None,
    ) -> str:
        if split is None:
            split = "training" if train is not False else "test"

        if split not in cls.VALID_SPLITS:
            valid = ", ".join(sorted(cls.VALID_SPLITS))
            raise ValueError(f"split doit appartenir à {{{valid}}}, reçu {split!r}.")

        return split

    @staticmethod
    def _build_processed_path(root: str | Path, split: str) -> Path:
        return Path(root) / "processed" / f"{split}.pt"

    @staticmethod
    def _load_processed_file(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed dataset file: {path}. "
                "Run data_mining.py to generate it."
            )

        loaded = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(loaded, (tuple, list)) and len(loaded) == 2:
            images, labels = loaded
        else:
            images = loaded
            labels = torch.zeros(images.shape[0], dtype=torch.int64)

        if not torch.is_tensor(images):
            images = torch.as_tensor(images)
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)

        return images, labels

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.images[index].float().div(255.0)
        img = (img - 0.5) * 2.0
        label = self.labels[index]
        return img, label


def data_loader(
    config,
    split: str = "training",
    shuffle: bool | None = None,
) -> DataLoader:
    """Construit le DataLoader standard historique."""
    if shuffle is None:
        shuffle = split == "training"

    dataset = ProcessedDataset(
        config.store_path_dataset,
        split=split,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
    )
