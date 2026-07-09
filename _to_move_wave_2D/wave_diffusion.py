# %%
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from lib.diffusion_lib.ImageVisualizer import ImageVisualizer
from lib.diffusion_lib.Logger import Logger
from lib.diffusion_lib.UNet import UNet
from lib.diffusion_lib.utils import get_best_device
from lib.diffusion_lib.ConditionalDDPM import WaveletConditionalDDPM
from lib.diffusion_lib.training_loop import *


@dataclass(frozen=True)
class Config:
    device: torch.device = None

    # Parametres generaux
    seed: int = 0
    store_path_dataset: str = "data/RT64"
    wavelet_level: int = 1
    viz: ImageVisualizer = ImageVisualizer(output_dir="outputs/img/wave")
    batch_size: int = 128
    do_train: bool = False
    n_epochs: int = 1000
    lr: float = 1e-3
    store_path: str = "outputs/model/wave_j1_RT64.pt"
    input_path: str = ""
    log_path: str = "outputs/logs/wave_j1_RT64.log"
    csv_path: str = "outputs/logs/wave_j1_RT64.csv"
    config_path: str = "outputs/model/wave_j1_RT64_config.json"

    # Parametres du DDPM conditionnel wavelet
    time_emb_dim: int = 100
    n_steps: int = 1000
    min_beta: float = 1e-4
    max_beta: float = 0.02

    # U-Net: entree = prior cA + details bruites, sortie = bruit sur les details.
    prior_channels: int = 1
    target_channels: int = 3
    unet_depth: int = 2
    unet_blocks_per_level: int = 2
    unet_base_channels: int = 10

    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        object.__setattr__(self, "device", get_best_device())
        print(self.device)

    @property
    def input_channels(self):
        return self.prior_channels + self.target_channels

    @property
    def train_path(self):
        return Path(self.store_path_dataset) / "processed" / f"j{self.wavelet_level}_training.pt"

    @property
    def val_path(self):
        return Path(self.store_path_dataset) / "processed" / f"j{self.wavelet_level}_validation.pt"

    @property
    def test_path(self):
        return Path(self.store_path_dataset) / "processed" / f"j{self.wavelet_level}_test.pt"





def load_wave_tensor(path, expected_channels=4):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset wavelet introuvable: {path}. Lance wave_data_mining_CEA.py "
            f"pour generer les fichiers j*_training.pt / validation.pt / test.pt."
        )

    data = torch.load(path, map_location="cpu").float()
    if data.ndim != 4:
        raise ValueError(f"Dataset attendu en (N, C, H, W), recu {tuple(data.shape)}.")
    if data.shape[1] != expected_channels:
        raise ValueError(f"Dataset attendu avec {expected_channels} canaux, recu {data.shape[1]}.")
    return data


def channel_stats(data):
    mean = data.mean(dim=(0, 2, 3))
    std = data.std(dim=(0, 2, 3)).clamp_min(1e-6)
    return mean, std


def normalize_with_stats(data, mean, std):
    return (data - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)


def make_loader(data, batch_size, shuffle):
    return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def show_wave_channels(viz, coeffs, title):
    """
    Sauvegarde une grille simple: chaque coefficient devient une image affichee.
    Utile pour verifier cA/cH/cV/cD sans modifier ImageVisualizer.
    """
    n = min(8, len(coeffs))
    images = coeffs[:n].reshape(n * coeffs.shape[1], 1, coeffs.shape[2], coeffs.shape[3])
    viz.show_images(images, title)


def build_model(config, image_hw, coeff_mean, coeff_std):
    network = UNet(
        n_steps=config.n_steps,
        time_emb_dim=config.time_emb_dim,
        size=image_hw[0],
        in_channels=config.input_channels,
        out_channels=config.target_channels,
        depth=config.unet_depth,
        blocks_per_level=config.unet_blocks_per_level,
        base_channels=config.unet_base_channels,
    )
    return WaveletConditionalDDPM(
        network,
        n_steps=config.n_steps,
        min_beta=config.min_beta,
        max_beta=config.max_beta,
        device=config.device,
        prior_channels=config.prior_channels,
        target_channels=config.target_channels,
        image_hw=image_hw,
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
    ).to(config.device)


# %%
def main():
    config = Config()
    logger = Logger(config.log_path, config.csv_path)

    expected_channels = config.prior_channels + config.target_channels 
    train_raw = load_wave_tensor(config.train_path, expected_channels=expected_channels)
    val_raw = load_wave_tensor(config.val_path, expected_channels=expected_channels)

    coeff_mean, coeff_std = channel_stats(train_raw)
    train_data = normalize_with_stats(train_raw, coeff_mean, coeff_std)
    val_data = normalize_with_stats(val_raw, coeff_mean, coeff_std)

    image_hw = tuple(train_data.shape[-2:])
    if image_hw[0] != image_hw[1]:
        raise ValueError(f"UNet attend des coefficients carres, recu HxW={image_hw}.")

    experiment_config = {
        "seed": config.seed,
        "store_path_dataset": config.store_path_dataset,
        "wavelet_level": config.wavelet_level,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "lr": config.lr,
        "store_path": config.store_path,
        "input_path": config.input_path,
        "time_emb_dim": config.time_emb_dim,
        "n_steps": config.n_steps,
        "min_beta": config.min_beta,
        "max_beta": config.max_beta,
        "coeff_chw": (expected_channels, *image_hw),
        "prior_channels": config.prior_channels,
        "target_channels": config.target_channels,
        "unet_depth": config.unet_depth,
        "unet_blocks_per_level": config.unet_blocks_per_level,
        "unet_base_channels": config.unet_base_channels,
        "coeff_mean": coeff_mean.tolist(),
        "coeff_std": coeff_std.tolist(),
        "device": config.device,
    }
    logger.log_experiment_start(experiment_config)
    logger.save_config(experiment_config, config.config_path)

    train_loader = make_loader(train_data, config.batch_size, shuffle=True)
    val_loader = make_loader(val_data, config.batch_size, shuffle=False)

    show_wave_channels(config.viz, train_data, f"wave_j{config.wavelet_level}_first_batch_normalized")

    ddpm = build_model(config, image_hw, coeff_mean, coeff_std)
    if config.input_path:
        input_path = Path(config.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {input_path}")
        ddpm.load_state_dict(torch.load(input_path, map_location=config.device))

    optimizer = Adam(ddpm.parameters(), lr=config.lr)

    if config.do_train:
        wave_training_loop(
            ddpm,
            train_loader,
            config.n_epochs,
            optimizer,
            config.device,
            store_path=config.store_path,
            logger=logger,
            val_loader=val_loader,
        )

    if Path(config.store_path).exists():
        ddpm.load_state_dict(torch.load(config.store_path, map_location=config.device))
    ddpm.eval()

    with torch.no_grad():
        prior, real_details = split_wave_batch(next(iter(val_loader)), config.device, config.prior_channels)
        generated = ddpm.sample(prior[:8], device=config.device)
        real_coeffs = torch.cat((prior[:8], real_details[:8]), dim=1)

    show_wave_channels(config.viz, real_coeffs.cpu(), f"wave_j{config.wavelet_level}_real_coeffs")
    show_wave_channels(config.viz, generated.cpu(), f"wave_j{config.wavelet_level}_generated_coeffs")

if __name__ == "__main__":
    main()
