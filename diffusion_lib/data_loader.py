from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class ProcessedDataset(Dataset):
    """
    Jeu de données PyTorch basé sur des fichiers déjà prétraités.

    Les images sont chargées depuis un fichier ``.pt`` contenant les tenseurs
    d'images et d'étiquettes, puis normalisées dans l'intervalle ``[-1, 1]``.
    """

    def __init__(self, root: str, train: bool | None = None, split: str | None = None):
        """
        Charge le jeu de données prétraité.

        Args:
            root (str): Répertoire racine du jeu de données.
            train (bool | None, optional): Ancienne API. Si True, charge
                ``training``; si False, charge ``test``.
            split (str | None, optional): Partition explicite à charger parmi
                ``training``, ``validation`` ou ``test``. Si fourni, remplace
                ``train``.

        Raises:
            FileNotFoundError: Si le fichier ``training.pt`` ou ``test.pt`` est
                introuvable.
        """
        if split is None:
            split = "training" if train is not False else "test"
        if split not in {"training", "validation", "test"}:
            raise ValueError("split doit etre 'training', 'validation' ou 'test'.")

        processed_path = Path(root) / "processed" / f"{split}.pt"
        if not processed_path.exists():
            raise FileNotFoundError(
                f"Missing processed dataset file: {processed_path}. "
                "Run data_mining.py to generate it."
            )

        loaded = torch.load(processed_path)
        if isinstance(loaded, (tuple, list)) and len(loaded) == 2:
            self.images, self.labels = loaded
        else:
            self.images = loaded
            self.labels = torch.zeros(self.images.shape[0], dtype=torch.int64)

        if self.images.ndim == 3:
            self.images = self.images.unsqueeze(1)

    def __len__(self) -> int:
        """
        Retourne le nombre d'exemples disponibles.

        Returns:
            int: Nombre d'éléments du jeu de données.
        """
        return self.images.shape[0]

    def __getitem__(self, index: int):
        """
        Récupère un exemple du jeu de données.

        Args:
            index (int): Indice de l'exemple à récupérer.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Une image normalisée et son label.
        """
        img = self.images[index].float().div(255.0)
        img = (img - 0.5) * 2
        label = self.labels[index]
        return img, label


def data_loader(config, split: str = "training", shuffle: bool | None = None):
    """
    Construit un DataLoader pour une partition prétraitée.

    Args:
        config: Objet de configuration contenant au moins ``store_path_dataset``
            et ``batch_size``.

    Returns:
        torch.utils.data.DataLoader: Chargeur de données prêt pour l'entraînement.
    """
    if shuffle is None:
        shuffle = split == "training"

    dataset = ProcessedDataset(config.store_path_dataset, split=split)
    loader = DataLoader(dataset, config.batch_size, shuffle=shuffle)
    return loader
