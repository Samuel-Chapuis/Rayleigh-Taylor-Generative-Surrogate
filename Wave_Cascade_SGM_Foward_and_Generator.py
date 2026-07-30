"""Train and generate a conditional VP-SGM wavelet cascade."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pywt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lib.diffusion_lib.SGM import SGM
from lib.diffusion_lib.UNet import UNet
from lib.wavelet_diffusion_lib.DDPM import DDPM
from lib.wavelet_diffusion_lib.ConditionalSGM import WaveletConditionalSGM
from lib.wavelet_diffusion_lib.ImageVisualizer import ImageVisualizer
from lib.wavelet_diffusion_lib.Logger import Logger
from lib.wavelet_diffusion_lib.training_loop import (
    split_wave_batch,
    training_loop,
    wave_sgm_training_loop,
)
from lib.wavelet_diffusion_lib.data_loader import wavelet_approximation_data_loader
from lib.wavelet_diffusion_lib.utils import get_best_device
from lib.wavelet_diffusion_lib.wavelet_utils import (
    channel_stats,
    load_wave_tensor,
    make_loader,
    normalize_with_stats,
    show_wave_channels,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_CONFIG_PATH = PROJECT_ROOT / "Wave_Cascade_Config_SGM.json"


@dataclass
class LevelConfig:
    wavelet_level: int
    device: torch.device
    seed: int = 0
    store_path_dataset: str = "data/RT64"
    batch_size: int = 128
    n_epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 1e-6
    grad_clip: float | None = 1.0
    input_path: str = ""
    time_emb_dim: int = 128
    beta_min: float = 0.1
    beta_max: float = 20.0
    eps_time: float = 1e-2
    sampling_steps: int = 256
    prediction_type: str = "v"
    sampler: str = "heun"
    prior_channels: int = 1
    target_channels: int = 3
    unet_depth: int = 3
    unet_blocks_per_level: int = 3
    unet_base_channels: int = 32
    model_dir: str = "saved_model/cascade_sgm"
    log_dir: str = "outputs/logs/cascade_sgm"
    image_dir: str = "outputs/img/wave_cascade_sgm"
    experiment_name: str = "RT64"

    @property
    def input_channels(self) -> int:
        return self.prior_channels + self.target_channels

    @property
    def train_path(self) -> Path:
        return Path(self.store_path_dataset) / "processed" / f"j{self.wavelet_level}_training.pt"

    @property
    def val_path(self) -> Path:
        return Path(self.store_path_dataset) / "processed" / f"j{self.wavelet_level}_validation.pt"

    @property
    def store_path(self) -> str:
        return str(Path(self.model_dir) / f"wave_j{self.wavelet_level}_{self.experiment_name}.pt")

    @property
    def saved_config_path(self) -> str:
        return str(Path(self.model_dir) / f"wave_j{self.wavelet_level}_{self.experiment_name}_config.json")

    @property
    def log_path(self) -> str:
        return str(Path(self.log_dir) / f"wave_j{self.wavelet_level}_{self.experiment_name}.log")

    @property
    def csv_path(self) -> str:
        return str(Path(self.log_dir) / f"wave_j{self.wavelet_level}_{self.experiment_name}.csv")


@dataclass
class CoarseConfig:
    """Configuration indépendante du DDPM inconditionnel sur cA."""

    wavelet_level: int
    device: torch.device
    seed: int = 0
    store_path_dataset: str = "data/RT64"
    batch_size: int = 128
    n_epochs: int = 100
    lr: float = 1e-3
    normalize_ca: bool = True
    do_train: bool = True
    input_path: str = ""
    store_path: str = "outputs/saved_models/cascade_sgm/coarse_cA3_RT64.pt"
    log_path: str = "outputs/logs/cascade_sgm/coarse_cA3_RT64.log"
    csv_path: str = "outputs/logs/cascade_sgm/coarse_cA3_RT64.csv"
    config_path: str = "outputs/saved_models/cascade_sgm/coarse_cA3_RT64_config.json"
    time_emb_dim: int = 100
    n_steps: int = 1000
    min_beta: float = 1e-4
    max_beta: float = 0.02
    image_channels: int = 1
    unet_depth: int = 3
    unet_blocks_per_level: int = 3
    unet_base_channels: int = 10
    unet_out_channels: int | None = None

    @property
    def train_path(self) -> Path:
        return Path(self.store_path_dataset) / "processed" / f"j{self.wavelet_level}_training.pt"

    @property
    def val_path(self) -> Path:
        return Path(self.store_path_dataset) / "processed" / f"j{self.wavelet_level}_validation.pt"


def absolute_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_run_config(path: Path = RUN_CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    levels = sorted({int(level) for level in config.get("levels", [])})
    if not levels or any(level < 1 for level in levels):
        raise ValueError("'levels' doit contenir des niveaux d'ondelettes positifs.")
    config["levels"] = levels
    config.setdefault("model", {})
    config.setdefault("train", {})
    config.setdefault("generate", {})
    config.setdefault("level_overrides", {})
    return config


def make_level_config(run_config: dict[str, Any], level: int, device: torch.device) -> LevelConfig:
    allowed = {field.name for field in fields(LevelConfig)} - {"wavelet_level", "device"}
    values: dict[str, Any] = {}
    for section_name in ("model", "train"):
        for key, value in run_config.get(section_name, {}).items():
            if key in allowed:
                values[key] = value
    for key, value in run_config.get("level_overrides", {}).get(str(level), {}).items():
        if key not in allowed:
            raise ValueError(f"Parametre inconnu pour j{level}: {key}")
        values[key] = value
    config = LevelConfig(wavelet_level=level, device=device, **values)
    seed_everything(config.seed + level)
    return config


def make_coarse_config(run_config: dict[str, Any], device: torch.device) -> CoarseConfig:
    """Construit la configuration coarse sans lire les paramètres SGM."""
    section = dict(run_config.get("coarse", {}))
    allowed = {field.name for field in fields(CoarseConfig)} - {"device"}
    control_keys = {"enabled", "preview_samples"}
    unknown = set(section) - allowed - control_keys
    if unknown:
        raise ValueError(f"Parametres coarse inconnus: {sorted(unknown)}")
    section = {key: value for key, value in section.items() if key in allowed}
    config = CoarseConfig(device=device, **section)
    seed_everything(config.seed)
    return config


def build_coarse_loaders(config: CoarseConfig):
    """Charge cA et partage les statistiques du train avec la validation."""
    train_loader = wavelet_approximation_data_loader(
        config, level=config.wavelet_level, split="training", shuffle=True,
        normalize=config.normalize_ca, return_label=True,
    )
    train_dataset = train_loader.dataset
    mean = train_dataset.mean if config.normalize_ca else None
    std = train_dataset.std if config.normalize_ca else None
    val_loader = wavelet_approximation_data_loader(
        config, level=config.wavelet_level, split="validation", shuffle=False,
        normalize=config.normalize_ca, mean=mean, std=std, return_label=True,
    )
    return train_loader, val_loader, mean, std


def build_coarse_ddpm(config: CoarseConfig, image_chw: tuple[int, int, int]) -> DDPM:
    network = UNet(
        n_steps=config.n_steps,
        time_emb_dim=config.time_emb_dim,
        size=image_chw[1],
        in_channels=image_chw[0],
        out_channels=config.unet_out_channels,
        depth=config.unet_depth,
        blocks_per_level=config.unet_blocks_per_level,
        base_channels=config.unet_base_channels,
    )
    return DDPM(
        network, n_steps=config.n_steps, min_beta=config.min_beta,
        max_beta=config.max_beta, device=config.device, image_chw=image_chw,
    ).to(config.device)


def train_coarse(config: CoarseConfig, preview_samples: int) -> None:
    """Entraîne le DDPM coarse, indépendamment des niveaux SGM conditionnels."""
    train_loader, val_loader, ca_mean, ca_std = build_coarse_loaders(config)
    first_images, _ = next(iter(train_loader))
    image_chw = tuple(first_images.shape[1:])
    if image_chw[0] != config.image_channels or image_chw[1] != image_chw[2]:
        raise ValueError(f"Forme cA incompatible avec le U-Net: {image_chw}")

    for path in (config.store_path, config.log_path, config.config_path):
        absolute_path(path).parent.mkdir(parents=True, exist_ok=True)
    experiment = {
        "kind": "unconditional_coarse_ddpm",
        "seed": config.seed, "store_path_dataset": config.store_path_dataset,
        "wavelet_level": config.wavelet_level, "normalize_ca": config.normalize_ca,
        "ca_mean": None if ca_mean is None else ca_mean.item(),
        "ca_std": None if ca_std is None else ca_std.item(),
        "batch_size": config.batch_size, "n_epochs": config.n_epochs, "lr": config.lr,
        "store_path": config.store_path, "input_path": config.input_path,
        "time_emb_dim": config.time_emb_dim, "n_steps": config.n_steps,
        "min_beta": config.min_beta, "max_beta": config.max_beta,
        "image_chw": image_chw, "unet_depth": config.unet_depth,
        "unet_blocks_per_level": config.unet_blocks_per_level,
        "unet_base_channels": config.unet_base_channels,
        "unet_out_channels": config.unet_out_channels,
    }
    logger = Logger(absolute_path(config.log_path), absolute_path(config.csv_path))
    logger.log_experiment_start(experiment)
    logger.save_config(experiment, absolute_path(config.config_path))

    ddpm = build_coarse_ddpm(config, image_chw)
    if config.input_path:
        checkpoint = absolute_path(config.input_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint coarse introuvable: {checkpoint}")
        ddpm.load_state_dict(torch.load(checkpoint, map_location=config.device, weights_only=True))
    optimizer = AdamW(ddpm.parameters(), lr=config.lr)
    if config.do_train:
        training_loop(
            ddpm, train_loader, config.n_epochs, optimizer, config.device,
            store_path=absolute_path(config.store_path), logger=logger,
            val_loader=val_loader,
        )
    checkpoint = absolute_path(config.store_path)
    if not config.do_train and config.input_path:
        checkpoint = absolute_path(config.input_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint coarse absent: {checkpoint}. Activez coarse.do_train ou renseignez coarse.input_path."
        )

    if preview_samples > 0:
        ddpm.load_state_dict(torch.load(checkpoint, map_location=config.device, weights_only=True))
        ddpm.eval()
        with torch.no_grad():
            generated = ddpm.sample(n_samples=preview_samples, device=config.device).cpu()
        if config.normalize_ca and ca_mean is not None and ca_std is not None:
            generated = generated * ca_std + ca_mean
        torch.save(generated, absolute_path(config.store_path).with_suffix(".preview.pt"))
    logger.log_experiment_end("Coarse DDPM training finished")


def build_sgm(config: LevelConfig | dict[str, Any], image_hw, coeff_mean, coeff_std, device):
    def get(name, default=None):
        return getattr(config, name, config.get(name, default) if isinstance(config, dict) else default)

    prior_channels = int(get("prior_channels", 1))
    target_channels = int(get("target_channels", 3))
    network = UNet(
        n_steps=1000,
        time_emb_dim=int(get("time_emb_dim", 128)),
        size=int(image_hw[0]),
        in_channels=prior_channels + target_channels,
        out_channels=target_channels,
        depth=int(get("unet_depth", 3)),
        blocks_per_level=int(get("unet_blocks_per_level", 3)),
        base_channels=int(get("unet_base_channels", 32)),
        continuous_time=True,
    )
    return WaveletConditionalSGM(
        network,
        beta_min=float(get("beta_min", 0.1)),
        beta_max=float(get("beta_max", 20.0)),
        eps_time=float(get("eps_time", 1e-2)),
        prediction_type=get("prediction_type", "v"),
        device=device,
        prior_channels=prior_channels,
        target_channels=target_channels,
        image_hw=image_hw,
        coeff_mean=torch.as_tensor(coeff_mean),
        coeff_std=torch.as_tensor(coeff_std),
    ).to(device)


def saved_config_dict(config, image_hw, coeff_mean, coeff_std):
    return {
        "seed": config.seed,
        "store_path_dataset": config.store_path_dataset,
        "wavelet_level": config.wavelet_level,
        "store_path": config.store_path,
        "time_emb_dim": config.time_emb_dim,
        "beta_min": config.beta_min,
        "beta_max": config.beta_max,
        "eps_time": config.eps_time,
        "sampling_steps": config.sampling_steps,
        "prediction_type": config.prediction_type,
        "sampler": config.sampler,
        "coeff_chw": (config.input_channels, *image_hw),
        "prior_channels": config.prior_channels,
        "target_channels": config.target_channels,
        "unet_depth": config.unet_depth,
        "unet_blocks_per_level": config.unet_blocks_per_level,
        "unet_base_channels": config.unet_base_channels,
        "coeff_mean": coeff_mean.tolist(),
        "coeff_std": coeff_std.tolist(),
    }


def train_one_level(config: LevelConfig, preview_samples: int) -> None:
    expected_channels = config.input_channels
    train_raw = load_wave_tensor(absolute_path(config.train_path), expected_channels=expected_channels)
    val_raw = load_wave_tensor(absolute_path(config.val_path), expected_channels=expected_channels)
    coeff_mean, coeff_std = channel_stats(train_raw)
    train_data = normalize_with_stats(train_raw, coeff_mean, coeff_std)
    val_data = normalize_with_stats(val_raw, coeff_mean, coeff_std)
    image_hw = tuple(train_data.shape[-2:])
    if image_hw[0] != image_hw[1]:
        raise ValueError(f"Les coefficients doivent etre carres, recu {image_hw}.")

    for path in (config.model_dir, config.log_dir, config.image_dir):
        absolute_path(path).mkdir(parents=True, exist_ok=True)
    saved = saved_config_dict(config, image_hw, coeff_mean, coeff_std)
    logger = Logger(absolute_path(config.log_path), absolute_path(config.csv_path))
    logger.log_experiment_start(saved)
    logger.save_config(saved, absolute_path(config.saved_config_path))
    train_loader = make_loader(train_data, config.batch_size, shuffle=True)
    val_loader = make_loader(val_data, config.batch_size, shuffle=False)
    show_wave_channels(ImageVisualizer(output_dir=absolute_path(config.image_dir)), train_data,
                       f"sgm_wave_j{config.wavelet_level}_train")

    sgm = build_sgm(config, image_hw, coeff_mean, coeff_std, config.device)
    if config.input_path:
        sgm.load_state_dict(torch.load(absolute_path(config.input_path), map_location=config.device, weights_only=True))
    optimizer = AdamW(sgm.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    wave_sgm_training_loop(sgm, train_loader, config.n_epochs, optimizer, config.device,
                           store_path=absolute_path(config.store_path), logger=logger,
                           val_loader=val_loader, grad_clip=config.grad_clip)
    sgm.load_state_dict(torch.load(absolute_path(config.store_path), map_location=config.device, weights_only=True))
    sgm.eval()
    with torch.no_grad():
        prior, real_details = split_wave_batch(next(iter(val_loader)), config.device, config.prior_channels)
        generated = sgm.sample(prior[:preview_samples], device=config.device,
                               n_steps=config.sampling_steps, solver=config.sampler)
    viz = ImageVisualizer(output_dir=absolute_path(config.image_dir))
    show_wave_channels(viz, torch.cat((prior[:preview_samples], real_details[:preview_samples]), 1).cpu(),
                       f"sgm_wave_j{config.wavelet_level}_real")
    show_wave_channels(viz, sgm.denormalize_coeffs(generated).cpu(),
                       f"sgm_wave_j{config.wavelet_level}_generated")


def load_saved_config(config: LevelConfig) -> dict[str, Any]:
    with absolute_path(config.saved_config_path).open("r", encoding="utf-8") as file:
        saved = json.load(file)
    saved["coeff_chw"] = tuple(saved["coeff_chw"])
    return saved


def load_model(saved: dict[str, Any], device: torch.device):
    model = build_sgm(saved, tuple(saved["coeff_chw"][1:]), saved["coeff_mean"], saved["coeff_std"], device)
    checkpoint = absolute_path(saved["store_path"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def generate_unconditional_ca(saved: dict[str, Any], device: torch.device, n_samples: int) -> torch.Tensor:
    """Generate cA with a standard (unconditional) SGM checkpoint."""
    image_chw = tuple(saved["image_chw"])
    network = UNet(
        n_steps=1000,
        time_emb_dim=int(saved.get("time_emb_dim", 128)),
        size=int(image_chw[1]),
        in_channels=int(image_chw[0]),
        out_channels=saved.get("unet_out_channels"),
        depth=int(saved.get("unet_depth", 3)),
        blocks_per_level=int(saved.get("unet_blocks_per_level", 3)),
        base_channels=int(saved.get("unet_base_channels", 32)),
        continuous_time=True,
    )
    model = SGM(
        network,
        beta_min=float(saved.get("beta_min", 0.1)),
        beta_max=float(saved.get("beta_max", 20.0)),
        eps_time=float(saved.get("eps_time", 1e-2)),
        prediction_type=saved.get("prediction_type", "v"),
        device=device,
        image_chw=image_chw,
    )
    checkpoint = absolute_path(saved["store_path"])
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        samples = model.sample(
            n_samples=n_samples,
            device=device,
            n_steps=int(saved.get("sampling_steps", 256)),
            solver=saved.get("sampler", "heun"),
            clip_denoised=bool(saved.get("clip_denoised", True)),
        ).cpu()
    if saved.get("normalize_ca", False):
        samples = samples * float(saved["ca_std"]) + float(saved["ca_mean"])
    return samples


def generate_coarse_ddpm(saved: dict[str, Any], device: torch.device, n_samples: int) -> torch.Tensor:
    """Génère cA avec le DDPM coarse entraîné par cette cascade."""
    image_chw = tuple(saved["image_chw"])
    config = type("SavedCoarseConfig", (), {
        "n_steps": int(saved["n_steps"]),
        "time_emb_dim": int(saved.get("time_emb_dim", 100)),
        "unet_out_channels": saved.get("unet_out_channels"),
        "unet_depth": int(saved.get("unet_depth", 3)),
        "unet_blocks_per_level": int(saved.get("unet_blocks_per_level", 3)),
        "unet_base_channels": int(saved.get("unet_base_channels", 10)),
        "min_beta": float(saved.get("min_beta", 1e-4)),
        "max_beta": float(saved.get("max_beta", 0.02)),
        "device": device,
    })()
    model = build_coarse_ddpm(config, image_chw)
    checkpoint = absolute_path(saved["store_path"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint coarse introuvable: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples=n_samples, device=device).cpu()
    if saved.get("normalize_ca", False):
        samples = samples * float(saved["ca_std"]) + float(saved["ca_mean"])
    return samples


def inverse_dwt_batch(ca, details, wavelet, mode):
    if ca.ndim != 4 or ca.shape[1] != 1 or details.ndim != 4 or details.shape[1] != 3:
        raise ValueError("IDWT attend cA=[N,1,H,W] et details=[N,3,H,W].")
    if ca.shape[0] != details.shape[0] or ca.shape[2:] != details.shape[2:]:
        raise ValueError(f"Dimensions incompatibles pour IDWT: {tuple(ca.shape)} / {tuple(details.shape)}")
    result = []
    for ca_i, details_i in zip(ca[:, 0].numpy(), details.numpy()):
        result.append(pywt.idwt2((ca_i, tuple(details_i)), wavelet=wavelet, mode=mode))
    return torch.from_numpy(np.stack(result)[:, None]).to(dtype=ca.dtype)


def generated_details(ca_physical, saved, model, device, batch_size):
    expected_hw = tuple(saved["coeff_chw"][1:])
    if tuple(ca_physical.shape[2:]) != expected_hw:
        raise ValueError(f"Resolution attendue {expected_hw}, recue {tuple(ca_physical.shape[2:])}.")
    batches = []
    mean = torch.as_tensor(saved["coeff_mean"], dtype=ca_physical.dtype)[0]
    std = torch.as_tensor(saved["coeff_std"], dtype=ca_physical.dtype)[0]
    if not torch.isfinite(std) or std <= 0:
        raise ValueError("Ecart-type invalide pour le canal cA.")
    loader = DataLoader(TensorDataset(ca_physical), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (ca_batch,) in loader:
            prior = ((ca_batch - mean) / std).to(device)
            sampled = model.sample(prior, device=device, n_steps=int(saved["sampling_steps"]),
                                   solver=saved.get("sampler", "heun"))
            detail_mean = torch.as_tensor(saved["coeff_mean"], dtype=sampled.dtype)[1:].view(1, 3, 1, 1)
            detail_std = torch.as_tensor(saved["coeff_std"], dtype=sampled.dtype)[1:].view(1, 3, 1, 1)
            batches.append((sampled[:, 1:].cpu() * detail_std + detail_mean).cpu())
    return torch.cat(batches)


def generate_cascade(run_config, device):
    generation = run_config["generate"]
    levels = sorted(run_config["levels"], reverse=True)
    if any(a - b != 1 for a, b in zip(levels, levels[1:])):
        raise ValueError(f"Niveaux non consecutifs pour la cascade: {levels}")
    batch_size = int(generation.get("batch_size", 64))
    n_samples = generation.get("n_samples")
    split = generation.get("dataset", "validation")
    wavelet = generation.get("wavelet", "db1")
    mode = generation.get("border_mode", "periodization")
    output_dir = absolute_path(generation.get("output_dir", "outputs/generated/cascade_sgm"))
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_source = generation.get("initial_ca_source", "dataset")

    # The initial cA can come directly from the requested dataset split. In
    # that mode no checkpoint/configuration is needed for the coarse level.
    coarsest = make_level_config(run_config, levels[0], device)
    coarse_data = load_wave_tensor(
        absolute_path(coarsest.store_path_dataset) / "processed" /
        f"j{levels[0]}_{split}.pt"
    )
    count = min(len(coarse_data), int(n_samples)) if n_samples is not None else min(len(coarse_data), batch_size)
    current_ca = coarse_data[:count, :1].float()
    if initial_source == "coarse_ddpm":
        coarse_config_path = absolute_path(
            generation.get("coarse_config_path", make_coarse_config(run_config, device).config_path)
        )
        with coarse_config_path.open("r", encoding="utf-8") as file:
            coarse_saved = json.load(file)
        if generation.get("coarse_checkpoint_path"):
            coarse_saved["store_path"] = generation["coarse_checkpoint_path"]
        current_ca = generate_coarse_ddpm(coarse_saved, device, count)
    elif initial_source == "generated":
        config_path = absolute_path(generation["initial_ca_config_path"])
        with config_path.open("r", encoding="utf-8") as file:
            initial_config = json.load(file)
        if generation.get("initial_ca_checkpoint_path"):
            initial_config["store_path"] = generation["initial_ca_checkpoint_path"]
        current_ca = generate_unconditional_ca(initial_config, device, count)
    elif initial_source != "dataset":
        raise ValueError("initial_ca_source doit valoir 'dataset', 'coarse_ddpm' ou 'generated'.")
    if tuple(current_ca.shape) != tuple(coarse_data[:count, :1].shape):
        raise ValueError("Le modele cA produit une resolution incompatible avec la cascade.")
    torch.save(current_ca, output_dir / f"j{levels[0]}_initial_cA.pt")

    for level in levels:
        level_config = make_level_config(run_config, level, device)
        saved = load_saved_config(level_config)
        model = load_model(saved, device)
        details = generated_details(current_ca, saved, model, device, batch_size)
        torch.save(details, output_dir / f"j{level}_generated_details.pt")
        torch.save(torch.cat((current_ca, details), dim=1), output_dir / f"j{level}_generated_coefficients.pt")
        current_ca = inverse_dwt_batch(current_ca, details, wavelet, mode)
        output_name = "generated_images.pt" if level == levels[-1] else f"j{level - 1}_reconstructed_cA.pt"
        torch.save(current_ca, output_dir / output_name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"j{level}: IDWT -> {tuple(current_ca.shape)}")
    return current_ca


def main():
    config = load_run_config()
    device = get_best_device()
    if config.get("coarse", {}).get("enabled", False):
        coarse_config = make_coarse_config(config, device)
        train_coarse(coarse_config, int(config["coarse"].get("preview_samples", 8)))
    if config["train"].get("enabled", False):
        for level in config["levels"]:
            level_config = make_level_config(config, level, device)
            train_one_level(level_config, int(config["train"].get("preview_samples", 8)))
    if config["generate"].get("enabled", False):
        generate_cascade(config, device)


if __name__ == "__main__":
    main()
