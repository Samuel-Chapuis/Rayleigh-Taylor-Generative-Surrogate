from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class ProcessedDataset(Dataset):
    """
    Jeu de données PyTorch basé sur des fichiers déjà prétraités.

    Les images sont chargées depuis un fichier ``.pt`` contenant les tenseurs
    d'images et d'étiquettes, puis normalisées dans l'intervalle ``[-1, 1]``.
    """

    def __init__(self, root: str, train: bool = True):
        """
        Charge le jeu de données prétraité.

        Args:
            root (str): Répertoire racine du jeu de données.
            train (bool, optional): Si True, charge la partition d'entraînement;
                sinon la partition de test. Par défaut True.

        Raises:
            FileNotFoundError: Si le fichier ``training.pt`` ou ``test.pt`` est
                introuvable.
        """
        split = "training" if train else "test"
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


def data_loader(config):
    """
    Construit le DataLoader d'entraînement.

    Args:
        config: Objet de configuration contenant au moins ``store_path_dataset``
            et ``batch_size``.

    Returns:
        torch.utils.data.DataLoader: Chargeur de données prêt pour l'entraînement.
    """
    dataset = ProcessedDataset(config.store_path_dataset, train=True)
    loader = DataLoader(dataset, config.batch_size, shuffle=True)
    return loader