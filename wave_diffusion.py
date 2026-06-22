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


class WaveletConditionalDDPM(nn.Module):
    """
    DDPM conditionnel pour coefficients d'ondelettes.

    Le canal 0 est le prior basse frequence cA. Les canaux 1..3 sont les details
    cH/cV/cD sur lesquels on applique la diffusion et dont le bruit est predit.
    """

    def __init__(
        self,
        network,
        n_steps=1000,
        min_beta=1e-4,
        max_beta=0.02,
        device=None,
        prior_channels=1,
        target_channels=3,
        image_hw=(32, 32),
        coeff_mean=None,
        coeff_std=None,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.device = device
        self.prior_channels = prior_channels
        self.target_channels = target_channels
        self.image_hw = image_hw
        self.network = network.to(device)

        self.register_buffer("betas", torch.linspace(min_beta, max_beta, n_steps))
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

        if coeff_mean is None:
            coeff_mean = torch.zeros(prior_channels + target_channels)
        if coeff_std is None:
            coeff_std = torch.ones(prior_channels + target_channels)
        self.register_buffer("coeff_mean", coeff_mean.float().reshape(1, -1, 1, 1))
        self.register_buffer("coeff_std", coeff_std.float().reshape(1, -1, 1, 1))

    def forward(self, details_0, t, eta=None):
        n, c, h, w = details_0.shape
        if c != self.target_channels:
            raise ValueError(f"Details attendus avec {self.target_channels} canaux, recu {c}.")

        if eta is None:
            eta = torch.randn(n, c, h, w, device=self.device)

        t = self._flatten_time(t)
        a_bar = self.alpha_bars[t].reshape(n, 1, 1, 1)
        return a_bar.sqrt() * details_0 + (1 - a_bar).sqrt() * eta

    def backward(self, noisy_details, t, prior):
        model_input = torch.cat((prior, noisy_details), dim=1)
        return self.network(model_input, t)

    def sample(self, prior, device=None):
        if device is None:
            device = self.device

        prior = prior.to(device)
        n, c, h, w = prior.shape
        if c != self.prior_channels:
            raise ValueError(f"Prior attendu avec {self.prior_channels} canaux, recu {c}.")

        details = torch.randn(n, self.target_channels, h, w, device=device)
        with torch.no_grad():
            for t in reversed(range(self.n_steps)):
                time_tensor = torch.full((n, 1), t, device=device, dtype=torch.long)
                eta_theta = self.backward(details, time_tensor, prior)

                alpha_t = self.alphas[t]
                alpha_t_bar = self.alpha_bars[t]
                details = (1 / alpha_t.sqrt()) * (
                    details - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
                )

                if t > 0:
                    details = details + self.betas[t].sqrt() * torch.randn_like(details)

        return torch.cat((prior, details), dim=1)

    def denormalize_coeffs(self, coeffs):
        return coeffs * self.coeff_std + self.coeff_mean

    def normalize_coeffs(self, coeffs):
        return (coeffs - self.coeff_mean) / self.coeff_std

    def _flatten_time(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        return t.to(self.device).long().reshape(-1)


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


def split_wave_batch(batch, device, prior_channels=1):
    coeffs = batch[0].to(device)
    prior = coeffs[:, :prior_channels]
    details = coeffs[:, prior_channels:]
    return prior, details


def wave_noise_prediction_loss(ddpm, batch, mse, device):
    prior, details = split_wave_batch(batch, device, prior_channels=ddpm.prior_channels)
    n = len(details)
    eta = torch.randn_like(details)
    t = torch.randint(0, ddpm.n_steps, (n,), device=device)
    noisy_details = ddpm(details, t, eta)
    eta_theta = ddpm.backward(noisy_details, t.reshape(n, -1), prior)
    return mse(eta_theta, eta)


def evaluate_loss(ddpm, loader, device):
    mse = nn.MSELoss()
    was_training = ddpm.training
    ddpm.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            loss = wave_noise_prediction_loss(ddpm, batch, mse, device)
            total_loss += loss.item() * len(batch[0]) / len(loader.dataset)

    if was_training:
        ddpm.train()
    return total_loss


def training_loop(ddpm, loader, n_epochs, optim, device, store_path, logger=None, val_loader=None):
    mse = nn.MSELoss()
    best_loss = float("inf")
    loss_list = []
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info(f"Wavelet conditional training started for {n_epochs} epochs")

    for epoch in tqdm(range(n_epochs), desc="Wavelet DDPM training", colour="#00ff00"):
        epoch_loss = 0.0
        for batch in tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500"):
            ddpm.train()
            loss = wave_noise_prediction_loss(ddpm, batch, mse, device)
            loss_value = loss.item()
            loss_list.append(loss_value)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            epoch_loss += loss_value * len(batch[0]) / len(loader.dataset)

        val_loss = evaluate_loss(ddpm, val_loader, device) if val_loader is not None else None
        selection_loss = val_loss if val_loss is not None else epoch_loss

        log_string = f"Train loss at epoch {epoch + 1}: {epoch_loss:.4f}"
        if val_loss is not None:
            log_string += f" | Val loss: {val_loss:.4f}"

        stored = best_loss > selection_loss
        if stored:
            best_loss = selection_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)
        if logger:
            metrics = {"train_loss": epoch_loss, "best_loss": best_loss, "stored": stored}
            if val_loss is not None:
                metrics["val_loss"] = val_loss
            logger.log_epoch(epoch + 1, metrics)
            logger.info(log_string)

    if logger:
        logger.log_experiment_end("Wavelet conditional training finished")

    return loss_list


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
