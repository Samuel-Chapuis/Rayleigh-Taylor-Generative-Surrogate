from dataclasses import dataclass
from pathlib import Path
import os
import random

import numpy as np
import torch
from torch.optim import Adam

from lib.diffusion_lib.ImageVisualizer import ImageVisualizer
from lib.diffusion_lib.Logger import Logger
from lib.diffusion_lib.data_loader import wavelet_approximation_data_loader
from lib.diffusion_lib.utils import get_best_device
from lib.diffusion_lib.DDPM import DDPM
from lib.diffusion_lib.UNet import UNet
from lib.diffusion_lib.training_loop import training_loop


@dataclass(frozen=True)
class Config:
    device: torch.device | None = None

    # Dataset wavelet
    seed: int = 0
    store_path_dataset: str = "data/RT64"
    wavelet_level: int = 3
    normalize_ca: bool = True

    # Entraînement
    viz: ImageVisualizer = ImageVisualizer(output_dir="outputs/img/ca3")
    batch_size: int = 128
    do_train: bool = True
    n_epochs: int = 1
    lr: float = 1e-3

    store_path: str = "outputs/model/ca3_RT64.pt"
    input_path: str = ""
    log_path: str = "outputs/logs/ca3_RT64.log"
    csv_path: str = "outputs/logs/ca3_RT64.csv"
    config_path: str = "outputs/model/ca3_RT64_config.json"

    # DDPM
    time_emb_dim: int = 100
    n_steps: int = 1000
    min_beta: float = 1e-4
    max_beta: float = 0.02

    # La résolution est déduite du dataset cA3.
    image_channels: int = 1
    unet_depth: int = 3
    unet_blocks_per_level: int = 3
    unet_base_channels: int = 10
    unet_out_channels: int | None = None

    def __post_init__(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        object.__setattr__(self, "device", get_best_device())
        print("Device:", self.device)



def build_ca_loaders(config: Config):
    """Construit les loaders cA3 en partageant les statistiques du train."""
    train_loader = wavelet_approximation_data_loader(
        config,
        level=config.wavelet_level,
        split="training",
        shuffle=True,
        normalize=config.normalize_ca,
        return_label=True,
    )

    train_dataset = train_loader.dataset
    ca_mean = train_dataset.mean if config.normalize_ca else None
    ca_std = train_dataset.std if config.normalize_ca else None

    val_loader = wavelet_approximation_data_loader(
        config,
        level=config.wavelet_level,
        split="validation",
        shuffle=False,
        normalize=config.normalize_ca,
        mean=ca_mean,
        std=ca_std,
        return_label=True,
    )

    return train_loader, val_loader, ca_mean, ca_std



def build_ddpm(config: Config, image_chw: tuple[int, int, int]) -> DDPM:
    return DDPM(
        UNet(
            n_steps=config.n_steps,
            time_emb_dim=config.time_emb_dim,
            size=image_chw[1],
            in_channels=image_chw[0],
            out_channels=config.unet_out_channels,
            depth=config.unet_depth,
            blocks_per_level=config.unet_blocks_per_level,
            base_channels=config.unet_base_channels,
        ),
        n_steps=config.n_steps,
        min_beta=config.min_beta,
        max_beta=config.max_beta,
        device=config.device,
        image_chw=image_chw,
    )



def main() -> None:
    config = Config()

    Path(config.store_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.config_path).parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, ca_mean, ca_std = build_ca_loaders(config)

    # return_label=True maintient l'interface historique : batch = (images, labels).
    first_images, first_labels = next(iter(train_loader))
    image_chw = tuple(first_images.shape[1:])

    if image_chw[0] != config.image_channels:
        raise ValueError(
            f"cA{config.wavelet_level} devrait avoir {config.image_channels} canal, "
            f"mais le loader retourne {image_chw}."
        )
    if image_chw[1] != image_chw[2]:
        raise ValueError(f"Le U-Net attend des champs carrés, reçu {image_chw}.")

    print(f"Train cA{config.wavelet_level}: {len(train_loader.dataset)} échantillons")
    print(f"Validation cA{config.wavelet_level}: {len(val_loader.dataset)} échantillons")
    print("Batch images:", tuple(first_images.shape))
    print("Batch labels:", tuple(first_labels.shape))
    print("image_chw déduit:", image_chw)
    if config.normalize_ca:
        print(f"Statistiques train cA: mean={ca_mean.item():.8g}, std={ca_std.item():.8g}")

    logger = Logger(config.log_path, config.csv_path)
    experiment_config = {
        "seed": config.seed,
        "store_path_dataset": config.store_path_dataset,
        "wavelet_level": config.wavelet_level,
        "loaded_channel": "cA",
        "normalize_ca": config.normalize_ca,
        "ca_mean": None if ca_mean is None else ca_mean.item(),
        "ca_std": None if ca_std is None else ca_std.item(),
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "lr": config.lr,
        "store_path": config.store_path,
        "input_path": config.input_path,
        "time_emb_dim": config.time_emb_dim,
        "n_steps": config.n_steps,
        "min_beta": config.min_beta,
        "max_beta": config.max_beta,
        "image_chw": image_chw,
        "unet_depth": config.unet_depth,
        "unet_blocks_per_level": config.unet_blocks_per_level,
        "unet_base_channels": config.unet_base_channels,
        "unet_out_channels": config.unet_out_channels,
        "device": str(config.device),
    }
    logger.log_experiment_start(experiment_config)
    logger.save_config(experiment_config, config.config_path)

    # Le visualiseur reçoit toujours (images, labels), comme avec ProcessedDataset.
    config.viz.show_first_batch(train_loader)

    ddpm = build_ddpm(config, image_chw).to(config.device)

    if config.input_path:
        input_path = Path(config.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {input_path}")
        ddpm.load_state_dict(torch.load(input_path, map_location=config.device))

    config.viz.show_forward(ddpm, train_loader, config.device)

    with torch.no_grad():
        generated_before = ddpm.sample()
    config.viz.show_images(generated_before, "ca3_before_training")

    optimizer = Adam(ddpm.parameters(), lr=config.lr)

    if config.do_train:
        training_loop(
            ddpm,
            train_loader,
            config.n_epochs,
            optimizer,
            config.device,
            store_path=config.store_path,
            logger=logger,
            val_loader=val_loader,
        )

    checkpoint_path = Path(config.store_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Aucun meilleur modèle trouvé dans {checkpoint_path}. "
            "Active do_train=True ou renseigne input_path/store_path avec un checkpoint existant."
        )

    best_model = build_ddpm(config, image_chw).to(config.device)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    best_model.eval()

    config.viz.show_backward(best_model, config.device)


if __name__ == "__main__":
    main()
