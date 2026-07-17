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


class WaveletApproximationDataset(ProcessedDataset):
    """
    Dataset pour les coefficients d'ondelettes d'un niveau donné.

    Structure attendue::

        root/
            processed/
                j{level}_training.pt
                j{level}_validation.pt
                j{level}_test.pt

    Chaque fichier doit contenir un tenseur de forme ``[N, C, H, W]`` selon la
    convention::

        channel 0 : cA
        channel 1 : cH
        channel 2 : cV
        channel 3 : cD

    Cette classe ne conserve que le coefficient d'approximation ``cA`` et
    retourne donc un tenseur de forme ``[1, H, W]`` pour chaque échantillon.

    Contrairement à ``ProcessedDataset``, aucune division par 255 ni
    normalisation vers [-1, 1] n'est appliquée par défaut, car les coefficients
    d'ondelettes peuvent être signés et ne sont généralement pas des pixels
    uint8.
    """

    def __init__(
        self,
        root: str | Path,
        level: int,
        train: bool | None = None,
        split: str | None = None,
        *,
        normalize: bool = False,
        mean: float | torch.Tensor | None = None,
        std: float | torch.Tensor | None = None,
        return_label: bool = False,
    ) -> None:
        if not isinstance(level, int) or isinstance(level, bool) or level < 1:
            raise ValueError(f"level doit être un entier strictement positif, reçu {level!r}.")

        split = self._resolve_split(train=train, split=split)
        processed_path = self._build_wavelet_path(
            root=root,
            level=level,
            split=split,
        )

        self.level = level
        self.split = split
        self.processed_path = processed_path
        self.return_label = return_label
        self.normalize = normalize

        coefficients, labels = self._load_processed_file(processed_path)

        if coefficients.ndim == 3:
            # Tolérance pour un fichier qui contiendrait déjà uniquement cA :
            # [N, H, W] -> [N, 1, H, W].
            coefficients = coefficients.unsqueeze(1)

        if coefficients.ndim != 4:
            raise ValueError(
                f"Le fichier {processed_path} doit contenir un tenseur "
                f"[N,C,H,W], reçu {tuple(coefficients.shape)}."
            )

        if coefficients.shape[1] < 1:
            raise ValueError(
                f"Le fichier {processed_path} ne contient aucun canal."
            )

        # On conserve explicitement la dimension de canal : [N,1,H,W].
        self.images = coefficients[:, 0:1].float().contiguous()
        self.labels = labels

        if normalize:
            if mean is None:
                mean = self.images.mean()
            if std is None:
                std = self.images.std(unbiased=False)

            self.mean = torch.as_tensor(mean, dtype=self.images.dtype)
            self.std = torch.as_tensor(std, dtype=self.images.dtype)

            if self.mean.numel() != 1 or self.std.numel() != 1:
                raise ValueError("mean et std doivent être scalaires pour le canal cA.")
            if not torch.isfinite(self.std) or self.std.item() <= 0:
                raise ValueError(f"Écart-type invalide pour cA : {self.std.item()}.")
        else:
            self.mean = None
            self.std = None

    @staticmethod
    def _build_wavelet_path(
        root: str | Path,
        level: int,
        split: str,
    ) -> Path:
        return Path(root) / "processed" / f"j{level}_{split}.pt"

    def __getitem__(self, index: int):
        ca = self.images[index]

        if self.normalize:
            ca = (ca - self.mean) / self.std

        if self.return_label:
            return ca, self.labels[index]

        return ca



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



def wavelet_approximation_data_loader(
    config,
    level: int | None = None,
    split: str = "training",
    shuffle: bool | None = None,
    *,
    normalize: bool = False,
    mean: float | torch.Tensor | None = None,
    std: float | torch.Tensor | None = None,
    return_label: bool = False,
) -> DataLoader:
    """
    Construit un DataLoader ne retournant que ``cA`` pour un niveau donné.

    Le niveau peut être fourni explicitement avec ``level``. Lorsqu'il est
    omis, la fonction utilise ``config.wavelet_level``.

    Exemple::

        loader = wavelet_approximation_data_loader(
            config,
            level=3,
            split="validation",
            shuffle=False,
        )

        ca_batch = next(iter(loader))  # [B,1,H,W]
    """
    if level is None:
        if not hasattr(config, "wavelet_level"):
            raise AttributeError(
                "level n'a pas été fourni et config ne possède pas "
                "l'attribut wavelet_level."
            )
        level = int(config.wavelet_level)

    if shuffle is None:
        shuffle = split == "training"

    dataset = WaveletApproximationDataset(
        root=config.store_path_dataset,
        level=level,
        split=split,
        normalize=normalize,
        mean=mean,
        std=std,
        return_label=return_label,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
    )