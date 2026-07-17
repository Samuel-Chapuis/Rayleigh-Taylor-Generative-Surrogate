# %%
from dataclasses import dataclass
from logging import config
from pathlib import Path
import random

from networkx import config
import numpy as np
import torch
from torch.optim import Adam

from lib.diffusion_lib.ImageVisualizer import ImageVisualizer
from lib.diffusion_lib.Logger import Logger
from lib.diffusion_lib.utils import get_best_device

# Ces imports gardent les anciens points d'acces wave_diffusion.<fonction>.
from lib.diffusion_lib.schedules import (
    SCHEDULE_REFERENCE_STEPS,
    diffusion_steps_from_snr,
    get_diffusion_schedule,
    linear_beta_schedule,
    snr_from_alpha_bar,
)
from lib.diffusion_lib.training_loop import split_wave_batch, wave_training_loop
from lib.diffusion_lib.wavelet_utils import (
    build_wavelet_model,
    channel_stats,
    do_diffusion_until_snr,
    load_wave_tensor,
    make_loader,
    normalize_with_stats,
    show_wave_channels,
)
# %%

@dataclass(frozen=True)
class Config:
    device: torch.device = None

    wavelet_level: int = 2

    # Parametres generaux
    seed: int = 0
    store_path_dataset: str = "data/RT64"
    viz: ImageVisualizer = ImageVisualizer(output_dir="outputs/img/wave")
    batch_size: int = 128
    do_train: bool = False
    n_epochs: int = 1
    lr: float = 1e-3
    input_path: str = ""


    store_path: str = "" 
    log_path: str = ""
    csv_path: str = ""
    config_path: str = "" 

    # Parametres du DDPM conditionnel wavelet
    time_emb_dim: int = 100
    snr_threshold: float = 2
    min_beta: float = 1e-4
    max_beta: float = 0.02
    schedule_reference_steps: int = SCHEDULE_REFERENCE_STEPS

    # U-Net: entree = prior cA + details bruites, sortie = bruit sur les details.
    prior_channels: int = 1
    target_channels: int = 3
    unet_depth: int = 2
    unet_blocks_per_level: int = 2
    unet_base_channels: int = 10

    def __post_init__(self):
        object.__setattr__(
            self,
            "store_path",
            f"outputs/model/wave_j{self.wavelet_level}_RT64.pt",
        )
        object.__setattr__(
            self,
            "log_path",
            f"outputs/logs/wave_j{self.wavelet_level}_RT64.log",
        )
        object.__setattr__(
            self,
            "csv_path",
            f"outputs/logs/wave_j{self.wavelet_level}_RT64.csv",
        )
        object.__setattr__(
            self,
            "config_path",
            f"outputs/model/wave_j{self.wavelet_level}_RT64_config.json",
        )

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        object.__setattr__(self, "device", get_best_device())
        print(self.device)

    @property
    def input_channels(self):
        return self.prior_channels + self.target_channels

    @property
    def n_steps(self):
        n_steps, _ = diffusion_steps_from_snr(
            self.snr_threshold,
            min_beta=self.min_beta,
            max_beta=self.max_beta,
            reference_steps=self.schedule_reference_steps,
        )
        return n_steps

    @property
    def effective_max_beta(self):
        _, effective_max_beta = diffusion_steps_from_snr(
            self.snr_threshold,
            min_beta=self.min_beta,
            max_beta=self.max_beta,
            reference_steps=self.schedule_reference_steps,
        )
        return effective_max_beta

    @property
    def train_path(self):
        print(f"Training path: {Path(self.store_path_dataset) / 'processed' / f'j{self.wavelet_level}_training.pt'}")
        return Path(self.store_path_dataset) / "processed" / f"j{self.wavelet_level}_training.pt"
        

    @property
    def val_path(self):
        print(f"Validation path: {Path(self.store_path_dataset) / 'processed' / f'j{self.wavelet_level}_validation.pt'}")
        return Path(self.store_path_dataset) / "processed" / f"j{self.wavelet_level}_validation.pt"

    @property
    def test_path(self):
        print(f"Test path: {Path(self.store_path_dataset) / 'processed' / f'j{self.wavelet_level}_test.pt'}")
        return Path(self.store_path_dataset) / "processed" / f"j{self.wavelet_level}_test.pt"


# %%
def main():
    config = Config()
    print(
        "Diffusion schedule from SNR: "
        f"snr_threshold={config.snr_threshold}, "
        f"n_steps={config.n_steps}, "
        f"min_beta={config.min_beta:g}, "
        f"effective_max_beta={config.effective_max_beta:g} "
        f"(max_beta_limit={config.max_beta:g})"
    )
    logger = Logger(config.log_path, config.csv_path)

    expected_channels = config.prior_channels + config.target_channels
    train_raw = load_wave_tensor(config.train_path, expected_channels=expected_channels)
    val_raw = load_wave_tensor(config.val_path, expected_channels=expected_channels)

    print(f"Train data shape: {train_raw.shape}, Val data shape: {val_raw.shape}")

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
        "snr_threshold": config.snr_threshold,
        "n_steps": config.n_steps,
        "schedule_reference_steps": config.schedule_reference_steps,
        "min_beta": config.min_beta,
        "max_beta": config.effective_max_beta,
        "max_beta_limit": config.max_beta,
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

    ddpm = build_wavelet_model(config, image_hw, coeff_mean, coeff_std)
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
