"""Workflow d'entrainement image pour le VP-SDE score-based."""

from dataclasses import asdict, dataclass
import os
import random


import numpy as np
import torch
from torch.optim import AdamW

from lib.diffusion_lib.ImageVisualizer import ImageVisualizer
from lib.diffusion_lib.Logger import Logger
from lib.diffusion_lib.SGM import SGM
from lib.diffusion_lib.UNet import UNet
from lib.diffusion_lib.data_loader import data_loader
from lib.diffusion_lib.training_loop import sgm_training_loop
from lib.diffusion_lib.utils import get_best_device


@dataclass(frozen=True)
class Config:
    seed: int = 0
    store_path_dataset: str = "data/RT64"
    image_chw: tuple[int, int, int] = (1, 64, 64)
    batch_size: int = 128
    n_epochs: int = 1000
    lr: float = 2e-4
    weight_decay: float = 1e-6
    grad_clip: float | None = 1.0
    do_train: bool = False
    store_path: str = "outputs/model/RT64_sgm.pt"
    input_path: str = ""
    log_path: str = "outputs/logs/RT64_sgm.log"
    csv_path: str = "outputs/logs/RT64_sgm.csv"
    config_path: str = "outputs/model/RT64_sgm_config.json"

    # VP-SDE continu
    beta_min: float = 0.1
    beta_max: float = 20.0
    eps_time: float = 1e-2
    sampling_steps: int = 256 # Present uniquement pour la generation
    prediction_type: str = "v"
    sampler: str = "heun"
    clip_denoised: bool = True
    time_emb_dim: int = 128

    # U-Net commun avec le workflow DDPM
    unet_depth: int = 3
    unet_blocks_per_level: int = 3
    unet_base_channels: int = 32
    unet_out_channels: int | None = None

    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        object.__setattr__(self, "device", get_best_device())
        print(self.device)


def build_sgm(config):
    network = UNet(
        n_steps=1000,  # inutilise en mode continu, conserve l'API commune
        time_emb_dim=config.time_emb_dim,
        size=config.image_chw[1],
        in_channels=config.image_chw[0],
        out_channels=config.unet_out_channels,
        depth=config.unet_depth,
        blocks_per_level=config.unet_blocks_per_level,
        base_channels=config.unet_base_channels,
        continuous_time=True,
    )
    return SGM(
        network,
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        eps_time=config.eps_time,
        prediction_type=config.prediction_type,
        device=config.device,
        image_chw=config.image_chw,
    )


def main(config=None):
    config = Config() if config is None else config
    logger = Logger(config.log_path, config.csv_path)
    loader = data_loader(config)
    val_loader = data_loader(config, split="validation", shuffle=False)

    first_batch = next(iter(loader))[0]
    actual_image_chw = tuple(first_batch.shape[1:])
    if actual_image_chw != config.image_chw:
        raise ValueError(
            f"Incoherence dataset/modele: dataset={actual_image_chw}, "
            f"config.image_chw={config.image_chw}."
        )

    experiment_config = asdict(config)
    logger.log_experiment_start(experiment_config)
    logger.save_config(experiment_config, config.config_path)
    config.viz.show_first_batch(loader) if hasattr(config, "viz") else ImageVisualizer(output_dir="outputs/img").show_first_batch(loader)

    sgm = build_sgm(config)
    if config.input_path:
        if not os.path.exists(config.input_path):
            raise FileNotFoundError(f"Checkpoint introuvable: {config.input_path}")
        sgm.load_state_dict(torch.load(config.input_path, map_location=config.device, weights_only=True))

    optimizer = AdamW(sgm.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.do_train:
        sgm_training_loop(
            sgm, loader, config.n_epochs, optimizer, config.device,
            store_path=config.store_path, logger=logger, val_loader=val_loader,
            grad_clip=config.grad_clip,
        )

    if not os.path.exists(config.store_path):
        raise FileNotFoundError(
            f"Aucun checkpoint SGM trouve: {config.store_path}. "
            "Active config.do_train ou fournis config.input_path."
        )
    sgm.load_state_dict(torch.load(config.store_path, map_location=config.device, weights_only=True))
    sgm.eval()
    visualizer = ImageVisualizer(output_dir="outputs/img")
    visualizer.show_images(
        sgm.sample(
            device=config.device,
            n_steps=config.sampling_steps,
            solver=config.sampler,
            clip_denoised=config.clip_denoised,
        ),
        "SGM generated images",
    )
    return sgm


if __name__ == "__main__":
    main()
