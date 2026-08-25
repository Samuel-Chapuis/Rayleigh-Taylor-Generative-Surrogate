from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pywt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.wavelet_diffusion_lib.ConditionalDDPM import WaveletConditionalDDPM
from lib.diffusion_lib.DDPM import DDPM
from lib.diffusion_lib.ImageVisualizer import ImageVisualizer
from lib.diffusion_lib.Logger import Logger
from lib.diffusion_lib.UNet import UNet
from lib.diffusion_lib.schedules import (
    SCHEDULE_REFERENCE_STEPS,
    diffusion_steps_from_snr,
)
from lib.wavelet_diffusion_lib.training_loop import split_wave_batch, wave_training_loop
from lib.diffusion_lib.utils import get_best_device
from lib.wavelet_diffusion_lib.wavelet_utils import (
    build_wavelet_model,
    channel_stats,
    load_wave_tensor,
    make_loader,
    normalize_with_stats,
    show_wave_channels,
)

# Aucun argument CLI : modifier ce chemin ou le contenu du fichier JSON.
RUN_CONFIG_PATH = PROJECT_ROOT / "wave_cascade_config.json"


@dataclass
class LevelConfig:
    """Configuration effective d'un modèle associé à un niveau j."""

    wavelet_level: int
    device: torch.device

    seed: int = 0
    store_path_dataset: str = "data/RT64"
    batch_size: int = 128
    n_epochs: int = 1
    lr: float = 1e-3
    input_path: str = ""

    time_emb_dim: int = 100
    manual_n_steps: int | None = None
    snr_threshold: float = 2.0
    min_beta: float = 1e-4
    max_beta: float = 0.02
    schedule_reference_steps: int = SCHEDULE_REFERENCE_STEPS

    prior_channels: int = 1
    target_channels: int = 3
    unet_depth: int = 2
    unet_blocks_per_level: int = 2
    unet_base_channels: int = 10

    model_dir: str = "outputs/model"
    log_dir: str = "outputs/logs"
    image_dir: str = "outputs/img/wave"
    experiment_name: str = "RT64"

    @property
    def input_channels(self) -> int:
        return self.prior_channels + self.target_channels

    @property
    def n_steps(self) -> int:
        if self.manual_n_steps is not None:
            if int(self.manual_n_steps) < 1:
                raise ValueError(f"manual_n_steps doit être >= 1, reçu {self.manual_n_steps}.")
            return int(self.manual_n_steps)

        n_steps, _ = diffusion_steps_from_snr(
            self.snr_threshold,
            min_beta=self.min_beta,
            max_beta=self.max_beta,
            reference_steps=self.schedule_reference_steps,
        )
        return n_steps

    @property
    def effective_max_beta(self) -> float:
        if self.manual_n_steps is not None:
            return self.max_beta

        _, effective_max_beta = diffusion_steps_from_snr(
            self.snr_threshold,
            min_beta=self.min_beta,
            max_beta=self.max_beta,
            reference_steps=self.schedule_reference_steps,
        )
        return effective_max_beta

    @property
    def train_path(self) -> Path:
        return Path(self.store_path_dataset) / "processed" / f"j{self.wavelet_level}_training.pt"

    @property
    def val_path(self) -> Path:
        return Path(self.store_path_dataset) / "processed" / f"j{self.wavelet_level}_validation.pt"

    @property
    def test_path(self) -> Path:
        return Path(self.store_path_dataset) / "processed" / f"j{self.wavelet_level}_test.pt"

    @property
    def store_path(self) -> str:
        return str(
            Path(self.model_dir)
            / f"wave_j{self.wavelet_level}_{self.experiment_name}.pt"
        )

    @property
    def saved_config_path(self) -> str:
        return str(
            Path(self.model_dir)
            / f"wave_j{self.wavelet_level}_{self.experiment_name}_config.json"
        )

    @property
    def log_path(self) -> str:
        return str(
            Path(self.log_dir)
            / f"wave_j{self.wavelet_level}_{self.experiment_name}.log"
        )

    @property
    def csv_path(self) -> str:
        return str(
            Path(self.log_dir)
            / f"wave_j{self.wavelet_level}_{self.experiment_name}.csv"
        )

    @property
    def viz(self) -> ImageVisualizer:
        return ImageVisualizer(output_dir=self.image_dir)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def load_run_config(path: Path = RUN_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {path}\n"
            "Copier wave_cascade_config.json à la racine du projet."
        )

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    levels = [int(level) for level in config.get("levels", [])]
    if not levels:
        raise ValueError("La configuration doit contenir au moins un niveau dans 'levels'.")
    if len(levels) != len(set(levels)):
        raise ValueError(f"Les niveaux doivent être uniques, reçu : {levels}.")
    if any(level < 1 for level in levels):
        raise ValueError(f"Tous les niveaux doivent être >= 1, reçu : {levels}.")

    config["levels"] = sorted(levels)
    config.setdefault("train", {})
    config.setdefault("generate", {})
    config.setdefault("level_overrides", {})
    return config


def make_level_config(
    run_config: dict[str, Any],
    level: int,
    device: torch.device,
) -> LevelConfig:
    values: dict[str, Any] = {}
    allowed = {field.name for field in fields(LevelConfig)} - {"wavelet_level", "device"}

    # Les paramètres de "model" sont les valeurs communes à tous les niveaux.
    for section_name in ("model", "train"):
        section = run_config.get(section_name, {})
        for key, value in section.items():
            if key == "n_steps":
                values["manual_n_steps"] = value
                continue
            if key in allowed:
                values[key] = value

    # Une surcharge locale permet par exemple un batch_size différent à j1.
    override = run_config.get("level_overrides", {}).get(str(level), {})
    for key, value in override.items():
        if key == "n_steps":
            values["manual_n_steps"] = value
            continue
        if key not in allowed and key != "n_steps":
            raise ValueError(f"Paramètre de niveau inconnu pour j{level}: {key}")
        values[key] = value

    config = LevelConfig(wavelet_level=level, device=device, **values)
    seed_everything(config.seed + level)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Entraînement d'un niveau
# -----------------------------------------------------------------------------


def experiment_config_dict(
    config: LevelConfig,
    expected_channels: int,
    image_hw: tuple[int, int],
    coeff_mean: torch.Tensor,
    coeff_std: torch.Tensor,
) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "store_path_dataset": config.store_path_dataset,
        "wavelet_level": config.wavelet_level,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "lr": config.lr,
        "store_path": config.store_path,
        "input_path": config.input_path,
        "time_emb_dim": config.time_emb_dim,
        "schedule_mode": "manual" if config.manual_n_steps is not None else "snr",
        "manual_n_steps": config.manual_n_steps,
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
        "device": str(config.device),
        "experiment_name": config.experiment_name,
    }


def train_one_level(config: LevelConfig, preview_samples: int = 8) -> None:
    print(f"\n{'=' * 72}\nEntraînement du modèle j{config.wavelet_level}\n{'=' * 72}")
    print(
        "Diffusion schedule from SNR: "
        f"threshold={config.snr_threshold}, n_steps={config.n_steps}, "
        f"min_beta={config.min_beta:g}, "
        f"effective_max_beta={config.effective_max_beta:g}"
    )

    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.image_dir).mkdir(parents=True, exist_ok=True)

    logger = Logger(config.log_path, config.csv_path)
    expected_channels = config.input_channels

    train_raw = load_wave_tensor(config.train_path, expected_channels=expected_channels)
    val_raw = load_wave_tensor(config.val_path, expected_channels=expected_channels)
    print(f"Train shape={tuple(train_raw.shape)}, validation shape={tuple(val_raw.shape)}")

    coeff_mean, coeff_std = channel_stats(train_raw)
    train_data = normalize_with_stats(train_raw, coeff_mean, coeff_std)
    val_data = normalize_with_stats(val_raw, coeff_mean, coeff_std)

    image_hw = tuple(train_data.shape[-2:])
    if image_hw[0] != image_hw[1]:
        raise ValueError(f"UNet attend des coefficients carrés, reçu HxW={image_hw}.")

    saved_config = experiment_config_dict(
        config,
        expected_channels,
        image_hw,
        coeff_mean,
        coeff_std,
    )
    logger.log_experiment_start(saved_config)
    logger.save_config(saved_config, config.saved_config_path)

    train_loader = make_loader(train_data, config.batch_size, shuffle=True)
    val_loader = make_loader(val_data, config.batch_size, shuffle=False)
    show_wave_channels(
        config.viz,
        train_data,
        f"wave_j{config.wavelet_level}_first_batch_normalized",
    )

    ddpm = build_wavelet_model(config, image_hw, coeff_mean, coeff_std)

    if config.input_path:
        input_path = PROJECT_ROOT / config.input_path
        if not input_path.exists():
            raise FileNotFoundError(f"Checkpoint de reprise introuvable : {input_path}")
        ddpm.load_state_dict(torch.load(input_path, map_location=config.device))
        print(f"Checkpoint de reprise chargé : {input_path}")

    optimizer = Adam(ddpm.parameters(), lr=config.lr)
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

    checkpoint_path = PROJECT_ROOT / config.store_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"L'entraînement j{config.wavelet_level} n'a pas produit {checkpoint_path}."
        )

    ddpm.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    ddpm.eval()

    with torch.no_grad():
        prior, real_details = split_wave_batch(
            next(iter(val_loader)),
            config.device,
            config.prior_channels,
        )
        n_preview = min(preview_samples, prior.shape[0])
        generated = ddpm.sample(prior[:n_preview], device=config.device)
        real_coeffs = torch.cat((prior[:n_preview], real_details[:n_preview]), dim=1)

    show_wave_channels(
        config.viz,
        real_coeffs.cpu(),
        f"wave_j{config.wavelet_level}_real_coeffs",
    )
    show_wave_channels(
        config.viz,
        generated.cpu(),
        f"wave_j{config.wavelet_level}_generated_coeffs",
    )

    del optimizer, ddpm, train_loader, val_loader, train_data, val_data
    if config.device.type == "cuda":
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Génération en cascade
# -----------------------------------------------------------------------------


def load_saved_model_config(level_config: LevelConfig) -> dict[str, Any]:
    path = PROJECT_ROOT / level_config.saved_config_path
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration entraînée introuvable pour j{level_config.wavelet_level}: {path}"
        )
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    config["coeff_chw"] = tuple(config["coeff_chw"])

    checkpoint_path = PROJECT_ROOT / config["store_path"]
    if not checkpoint_path.exists():
        sibling_checkpoint = path.with_name(path.name.replace("_config.json", ".pt"))
        if sibling_checkpoint.exists():
            config["store_path"] = str(sibling_checkpoint.relative_to(PROJECT_ROOT))
    return config


def build_model_from_saved_config(
    config: dict[str, Any],
    device: torch.device,
) -> WaveletConditionalDDPM:
    coeff_chw = tuple(config["coeff_chw"])

    network = UNet(
        n_steps=int(config["n_steps"]),
        time_emb_dim=int(config["time_emb_dim"]),
        size=int(coeff_chw[1]),
        in_channels=int(config["prior_channels"]) + int(config["target_channels"]),
        out_channels=int(config["target_channels"]),
        depth=int(config["unet_depth"]),
        blocks_per_level=int(config["unet_blocks_per_level"]),
        base_channels=int(config["unet_base_channels"]),
    )

    ddpm = WaveletConditionalDDPM(
        network,
        n_steps=int(config["n_steps"]),
        min_beta=float(config["min_beta"]),
        max_beta=float(config["max_beta"]),
        device=device,
        prior_channels=int(config["prior_channels"]),
        target_channels=int(config["target_channels"]),
        image_hw=coeff_chw[1:],
        coeff_mean=torch.tensor(config["coeff_mean"], dtype=torch.float32),
        coeff_std=torch.tensor(config["coeff_std"], dtype=torch.float32),
    )

    checkpoint_path = PROJECT_ROOT / config["store_path"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    ddpm.load_state_dict(torch.load(checkpoint_path, map_location=device))
    ddpm.to(device)
    ddpm.eval()
    return ddpm


def load_unconditional_ca_config(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    path = PROJECT_ROOT / config_path
    if not path.exists():
        raise FileNotFoundError(f"Configuration du modèle cA non conditionné introuvable : {path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    config["image_chw"] = tuple(config["image_chw"])
    if int(config.get("image_chw", (0,))[0]) != 1:
        raise ValueError(
            "La cascade attend un modèle non conditionné générant un seul canal cA, "
            f"reçu image_chw={config['image_chw']}."
        )
    if checkpoint_path is not None:
        config["store_path"] = str(checkpoint_path)
    return config


def build_unconditional_ca_model(
    config: dict[str, Any],
    device: torch.device,
) -> DDPM:
    image_chw = tuple(config["image_chw"])
    network = UNet(
        n_steps=int(config["n_steps"]),
        time_emb_dim=int(config["time_emb_dim"]),
        size=int(image_chw[1]),
        in_channels=int(image_chw[0]),
        out_channels=config.get("unet_out_channels"),
        depth=int(config["unet_depth"]),
        blocks_per_level=int(config["unet_blocks_per_level"]),
        base_channels=int(config["unet_base_channels"]),
    )
    ddpm = DDPM(
        network,
        n_steps=int(config["n_steps"]),
        min_beta=float(config["min_beta"]),
        max_beta=float(config["max_beta"]),
        device=device,
        image_chw=image_chw,
    )

    checkpoint_path = PROJECT_ROOT / config["store_path"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint du modèle cA non conditionné introuvable : {checkpoint_path}")

    ddpm.load_state_dict(torch.load(checkpoint_path, map_location=device))
    ddpm.to(device)
    ddpm.eval()
    return ddpm


def denormalize_unconditional_ca(
    sampled_ca: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    sampled_ca = sampled_ca.detach().cpu().float()
    if not bool(config.get("normalize_ca", False)):
        return sampled_ca

    mean = config.get("ca_mean")
    std = config.get("ca_std")
    if mean is None or std is None:
        raise ValueError("normalize_ca=True mais ca_mean/ca_std sont absents de la config cA.")

    std_tensor = torch.tensor(float(std), dtype=sampled_ca.dtype)
    if not torch.isfinite(std_tensor) or std_tensor <= 0:
        raise ValueError(f"Écart-type cA invalide dans la config non conditionnée : {std}.")
    return sampled_ca * std_tensor + float(mean)


def generate_unconditional_ca(
    ca_config: dict[str, Any],
    device: torch.device,
    n_samples: int,
    batch_size: int,
) -> torch.Tensor:
    ddpm = build_unconditional_ca_model(ca_config, device)
    generated_batches = []

    with torch.no_grad():
        n_done = 0
        batch_index = 0
        while n_done < n_samples:
            current_batch_size = min(batch_size, n_samples - n_done)
            batch_index += 1
            print(
                f"cA non conditionné: génération batch {batch_index}, "
                f"{current_batch_size} échantillons"
            )
            sampled = ddpm.sample(n_samples=current_batch_size, device=device)
            generated_batches.append(denormalize_unconditional_ca(sampled, ca_config))
            n_done += current_batch_size

    generated_ca = torch.cat(generated_batches, dim=0)
    del ddpm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return generated_ca


def normalize_approximation(ca: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    if ca.ndim != 4 or ca.shape[1] != 1:
        raise ValueError(f"cA doit avoir la forme [N,1,H,W], reçu {tuple(ca.shape)}.")

    mean = torch.as_tensor(config["coeff_mean"], dtype=ca.dtype)[0]
    std = torch.as_tensor(config["coeff_std"], dtype=ca.dtype)[0]
    if not torch.isfinite(std) or std <= 0:
        raise ValueError(f"Écart-type cA invalide pour j{config['wavelet_level']}: {std}.")
    return (ca - mean) / std


def inverse_dwt_batch(
    ca: torch.Tensor,
    details: torch.Tensor,
    wavelet: str,
    mode: str,
) -> torch.Tensor:
    if ca.ndim != 4 or ca.shape[1] != 1:
        raise ValueError(f"cA attendu [N,1,H,W], reçu {tuple(ca.shape)}.")
    if details.ndim != 4 or details.shape[1] != 3:
        raise ValueError(f"Détails attendus [N,3,H,W], reçu {tuple(details.shape)}.")
    if ca.shape[0] != details.shape[0] or ca.shape[2:] != details.shape[2:]:
        raise ValueError(
            f"Dimensions incompatibles : cA={tuple(ca.shape)}, details={tuple(details.shape)}."
        )

    ca_np = ca.detach().cpu().numpy()
    details_np = details.detach().cpu().numpy()
    reconstructed = []

    for index in range(ca_np.shape[0]):
        c_a = ca_np[index, 0]
        c_h, c_v, c_d = details_np[index]
        reconstructed.append(
            pywt.idwt2((c_a, (c_h, c_v, c_d)), wavelet=wavelet, mode=mode)
        )

    output = np.stack(reconstructed, axis=0)[:, None]
    return torch.from_numpy(output).to(dtype=ca.dtype)


def extract_physical_details(
    sampled_normalized: torch.Tensor,
    ddpm: WaveletConditionalDDPM,
    ca_physical: torch.Tensor,
    level: int,
) -> torch.Tensor:
    sampled_physical = ddpm.denormalize_coeffs(sampled_normalized)
    if sampled_physical.ndim != 4:
        raise RuntimeError(
            f"j{level}: sortie inattendue {tuple(sampled_physical.shape)}; attendu [N,C,H,W]."
        )

    if sampled_physical.shape[1] == 3:
        return sampled_physical

    if sampled_physical.shape[1] == 4:
        returned_ca = sampled_physical[:, 0:1]
        if returned_ca.shape != ca_physical.shape:
            raise RuntimeError(
                f"j{level}: cA retourné {tuple(returned_ca.shape)} != "
                f"cA condition {tuple(ca_physical.shape)}."
            )
        max_difference = (returned_ca.cpu() - ca_physical.cpu()).abs().max().item()
        print(
            f"j{level}: sampler=[cA,cH,cV,cD], détails=canaux 1:4; "
            f"max|cA_returned-cA_input|={max_difference:.3e}"
        )
        return sampled_physical[:, 1:4]

    raise RuntimeError(
        f"j{level}: {sampled_physical.shape[1]} canaux retournés; attendu 3 ou 4."
    )


def generate_details_for_level(
    ca_physical: torch.Tensor,
    level: int,
    saved_config: dict[str, Any],
    ddpm: WaveletConditionalDDPM,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    expected_hw = tuple(saved_config["coeff_chw"][1:])
    if tuple(ca_physical.shape[2:]) != expected_hw:
        raise ValueError(
            f"j{level}: résolution attendue {expected_hw}, "
            f"cA reçu {tuple(ca_physical.shape[2:])}."
        )

    loader = DataLoader(
        TensorDataset(ca_physical),
        batch_size=batch_size,
        shuffle=False,
    )
    generated_batches = []

    with torch.no_grad():
        for batch_index, (ca_batch,) in enumerate(loader, start=1):
            prior = normalize_approximation(ca_batch, saved_config).to(device)
            print(
                f"j{level}: génération batch {batch_index}/{len(loader)}, "
                f"prior={tuple(prior.shape)}"
            )
            sampled_normalized = ddpm.sample(prior, device=device)
            details = extract_physical_details(
                sampled_normalized,
                ddpm,
                ca_batch,
                level,
            )
            generated_batches.append(details.cpu())

    return torch.cat(generated_batches, dim=0)


def generate_cascade(
    run_config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    generation = run_config["generate"]
    levels = sorted(run_config["levels"], reverse=True)

    if any(a - b != 1 for a, b in zip(levels, levels[1:])):
        raise ValueError(
            f"La cascade exige des niveaux consécutifs en ordre décroissant, reçu {levels}."
        )

    dataset_split = generation.get("dataset", "validation")
    batch_size = int(generation.get("batch_size", 64))
    n_batches = generation.get("n_batches", 1)
    n_samples_requested = generation.get("n_samples")
    wavelet = generation.get("wavelet", "db1")
    border_mode = generation.get("border_mode", "periodization")
    output_dir = PROJECT_ROOT / generation.get("output_dir", "outputs/generated/cascade")
    save_intermediate = bool(generation.get("save_intermediate", True))
    initial_ca_source = generation.get("initial_ca_source", "dataset")
    initial_ca_config_path = generation.get(
        "initial_ca_config_path",
        "outputs/model/ca3_RT64_config.json",
    )
    initial_ca_checkpoint_path = generation.get("initial_ca_checkpoint_path")
    output_dir.mkdir(parents=True, exist_ok=True)

    if initial_ca_source not in {"dataset", "generated"}:
        raise ValueError(
            "generate.initial_ca_source doit valoir 'dataset' ou 'generated', "
            f"reçu {initial_ca_source!r}."
        )

    export_generated_initial_ca_requested = generation.get("export_generated_initial_ca")
    if initial_ca_source == "dataset":
        export_generated_initial_ca = False
        if bool(export_generated_initial_ca_requested):
            print(
                "generate.export_generated_initial_ca=True ignoré car "
                "generate.initial_ca_source='dataset': aucun modèle cA non conditionné "
                "n'est requis."
            )
    else:
        export_generated_initial_ca = bool(
            True
            if export_generated_initial_ca_requested is None
            else export_generated_initial_ca_requested
        )

    coarsest_level = levels[0]
    coarsest_level_config = make_level_config(run_config, coarsest_level, device)
    coarsest_saved_config = load_saved_model_config(coarsest_level_config)
    dataset_path = (
        PROJECT_ROOT
        / coarsest_saved_config["store_path_dataset"]
        / "processed"
        / f"j{coarsest_level}_{dataset_split}.pt"
    )

    coarsest_coeffs = load_wave_tensor(dataset_path)
    if coarsest_coeffs.ndim != 4 or coarsest_coeffs.shape[1] < 1:
        raise ValueError(
            f"Dataset grossier attendu [N,C,H,W], reçu {tuple(coarsest_coeffs.shape)}."
        )

    if n_samples_requested is not None:
        n_samples = min(len(coarsest_coeffs), int(n_samples_requested))
    elif n_batches is None:
        n_samples = len(coarsest_coeffs)
    else:
        n_samples = min(len(coarsest_coeffs), batch_size * int(n_batches))

    if n_samples <= 0:
        raise ValueError("Aucun échantillon sélectionné pour la génération.")

    reference_ca = coarsest_coeffs[:n_samples, 0:1].float().cpu()
    if save_intermediate:
        torch.save(reference_ca, output_dir / f"j{coarsest_level}_reference_cA.pt")

    generated_initial_ca = None
    if initial_ca_source == "generated" or export_generated_initial_ca:
        ca_config = load_unconditional_ca_config(
            initial_ca_config_path,
            checkpoint_path=initial_ca_checkpoint_path,
        )
        expected_level = int(ca_config.get("wavelet_level", coarsest_level))
        if expected_level != coarsest_level:
            raise ValueError(
                f"Le modèle non conditionné génère cA{expected_level}, "
                f"mais la cascade démarre à j{coarsest_level}."
            )
        if tuple(ca_config["image_chw"][1:]) != tuple(reference_ca.shape[2:]):
            raise ValueError(
                "Résolution du modèle cA non conditionné incompatible : "
                f"config={tuple(ca_config['image_chw'][1:])}, "
                f"cascade={tuple(reference_ca.shape[2:])}."
            )

        generated_initial_ca = generate_unconditional_ca(
            ca_config,
            device=device,
            n_samples=n_samples,
            batch_size=batch_size,
        )
        if generated_initial_ca.shape != reference_ca.shape:
            raise RuntimeError(
                "Le modèle cA non conditionné a produit une forme incompatible : "
                f"{tuple(generated_initial_ca.shape)} != {tuple(reference_ca.shape)}."
            )

    if initial_ca_source == "generated":
        current_ca = generated_initial_ca
    else:
        current_ca = reference_ca

    if save_intermediate:
        if generated_initial_ca is not None:
            torch.save(generated_initial_ca, output_dir / f"j{coarsest_level}_generated_initial_cA.pt")
        torch.save(current_ca, output_dir / f"j{coarsest_level}_initial_cA.pt")

    print(
        f"\nDébut de la cascade {levels}: "
        f"cA initial={tuple(current_ca.shape)} source={initial_ca_source}"
    )

    for level in levels:
        print(f"\n{'-' * 72}\nGénération cascade j{level}\n{'-' * 72}")
        level_config = make_level_config(run_config, level, device)
        saved_config = load_saved_model_config(level_config)
        ddpm = build_model_from_saved_config(saved_config, device)

        details = generate_details_for_level(
            current_ca,
            level,
            saved_config,
            ddpm,
            device,
            batch_size,
        )
        full_coefficients = torch.cat((current_ca, details), dim=1)

        if save_intermediate:
            torch.save(details, output_dir / f"j{level}_generated_details.pt")
            torch.save(
                full_coefficients,
                output_dir / f"j{level}_generated_coefficients.pt",
            )

        current_ca = inverse_dwt_batch(
            current_ca,
            details,
            wavelet=wavelet,
            mode=border_mode,
        )

        output_name = (
            "generated_images.pt"
            if level == levels[-1]
            else f"j{level - 1}_reconstructed_cA.pt"
        )
        torch.save(current_ca, output_dir / output_name)
        print(f"Sortie IDWT : {tuple(current_ca.shape)} -> {output_name}")

        del ddpm, details, full_coefficients
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nImages finales : {tuple(current_ca.shape)}")
    print(f"Fichier : {output_dir / 'generated_images.pt'}")
    return current_ca


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def main() -> None:
    run_config = load_run_config()
    device = get_best_device()
    print(f"Device: {device}")
    print(f"Niveaux configurés: {run_config['levels']}")

    train_config = run_config["train"]
    if bool(train_config.get("enabled", False)):
        preview_samples = int(train_config.get("preview_samples", 8))
        for level in run_config["levels"]:
            level_config = make_level_config(run_config, level, device)
            train_one_level(level_config, preview_samples=preview_samples)
    else:
        print("Entraînement désactivé dans la configuration.")

    if bool(run_config["generate"].get("enabled", False)):
        generate_cascade(run_config, device)
    else:
        print("Génération en cascade désactivée dans la configuration.")


if __name__ == "__main__":
    main()
