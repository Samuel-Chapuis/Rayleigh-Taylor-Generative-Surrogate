"""Shared visualization and metric utilities for RT-Diffusion notebooks."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
from scipy.stats import skewnorm
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, TensorDataset


REPORT_PLOT_STYLE = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
}


@dataclass(frozen=True)
class BaselineConfig:
    data_root: Path
    generated_pt: Path
    noise_pt: Path
    mnist_pt: Path
    loss_csv: Path
    phyfid_encoder: Path
    phyfid_stats: Path
    phyfid_batch_size: int = 128
    fid_dims: int = 64
    fid_batch_size: int = 25
    cmap: str = "gray"


@dataclass(frozen=True)
class CascadeConfig:
    name: str
    output_dir: Path
    csv_dir: Path
    data_root: Path
    dataset: str = "validation"
    cascade_levels: tuple[int, ...] = (2, 1)
    wavelet: str = "db1"
    mode: str = "periodization"
    phyfid_root: Path | None = None
    phyfid_batch_size: int = 128
    phyfid_max_images: int = 1000
    cmap: str = "gray"
    coarse_source: str = "unknown"

    @property
    def is_reference_coarse(self) -> bool:
        return self.coarse_source == "dataset"

    @property
    def experiment_label(self) -> str:
        if self.coarse_source == "dataset":
            return "WSGM_reference"
        if self.coarse_source == "coarse_sgm":
            return "WSGM"
        return self.name

    @property
    def coarse_level(self) -> int:
        return max(self.cascade_levels)

    @property
    def coarse_key(self) -> int:
        return self.coarse_level + 1

    @property
    def real_coeff_paths(self) -> dict[int, Path]:
        return {
            level: self.data_root / f"j{level}_{self.dataset}.pt"
            for level in self.cascade_levels
        }

    @property
    def generated_coeff_paths(self) -> dict[int, Path]:
        return {
            self.coarse_key: self.output_dir / f"j{self.coarse_level}_initial_cA.pt",
            **{
                level: self.output_dir / f"j{level}_generated_coefficients.pt"
                for level in self.cascade_levels
            },
        }

    @property
    def reconstructed_ca_paths(self) -> dict[int, Path]:
        return {
            level - 1: self.output_dir / f"j{level - 1}_reconstructed_cA.pt"
            for level in self.cascade_levels
            if level > 1
        }

    @property
    def loss_paths(self) -> dict[int, Path]:
        return {
            self.coarse_key: self.csv_dir / f"coarse_cA{self.coarse_level}_RT64.csv",
            **{level: self.csv_dir / f"wave_j{level}_RT64.csv" for level in self.cascade_levels},
        }

    @property
    def final_generated_images_pt(self) -> Path:
        return self.output_dir / "generated_images.pt"

    @property
    def phyfid_encoder_64(self) -> Path:
        root = self.phyfid_root or self.output_dir.parents[1] / "phyFID"
        return root / "64phyfid_encoder.pt"

    @property
    def phyfid_encoder_coarse(self) -> Path:
        root = self.phyfid_root or self.output_dir.parents[1] / "phyFID"
        return root / f"coarse_cA{self.coarse_level}_phyfid_encoder.pt"


def find_project_root(start: Path | str | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "lib").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError("Cannot find the RT-Diffusion project root.")


def configure_project_paths(project_root: Path | str | None = None) -> Path:
    project_root = find_project_root(project_root)
    paths = [project_root, project_root / "lib" / "pytorch-fid-master" / "src"]
    for path in paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    return project_root


def apply_report_style() -> None:
    plt.rcParams.update(REPORT_PLOT_STYLE)


def default_baseline_config(project_root: Path) -> BaselineConfig:
    data_root = project_root / "data" / "RT64" / "processed"
    return BaselineConfig(
        data_root=data_root,
        generated_pt=project_root / "outputs" / "generated" / "64dataset_e-4_s1024.pt",
        noise_pt=project_root / "outputs" / "generated" / "64noise_dataset_sgm.pt",
        mnist_pt=project_root / "data" / "MNIST" / "processed" / "training.pt",
        loss_csv=project_root / "outputs" / "logs" / "RT64_sgm.csv",
        phyfid_encoder=project_root / "outputs" / "phyFID" / "64phyfid_encoder.pt",
        phyfid_stats=project_root / "outputs" / "phyFID" / "64phyfid_val_stats.npz",
    )


def default_cascade_configs(project_root: Path) -> dict[str, CascadeConfig]:
    data_root = project_root / "data" / "RT64" / "processed"
    generated_root = project_root / "outputs" / "generated"
    logs_root = project_root / "outputs" / "logs"
    phyfid_root = project_root / "outputs" / "phyFID"
    return {
        "generated_coarse": CascadeConfig(
            name="WSGM - cascade complete avec coarse genere",
            output_dir=generated_root / "cascade_sgm_hcirc_blurpool_eps00001",
            csv_dir=logs_root / "cascade_sgm_hcirc_blurpool_eps00001",
            data_root=data_root,
            phyfid_root=phyfid_root,
            coarse_source="coarse_sgm",
        ),
        "reference_coarse": CascadeConfig(
            name="WSGM_reference - coarse reel injecte depuis le dataset",
            output_dir=generated_root / "cascade_sgm_1024",
            csv_dir=logs_root / "cascade_sgm_hcirc_blurpool",
            data_root=data_root,
            phyfid_root=phyfid_root,
            coarse_source="dataset",
        ),
    }


def load_pt_dataset(path: Path | str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ensure_tensor(value) -> torch.Tensor:
    return value if torch.is_tensor(value) else torch.as_tensor(value)


def squeeze_channel(images) -> torch.Tensor:
    images = ensure_tensor(images)
    if images.ndim == 4 and images.shape[1] == 1:
        return images[:, 0]
    return images


def to_numpy_images(images) -> np.ndarray:
    return squeeze_channel(images).detach().cpu().numpy().astype(np.float32)


def as_txy_numpy(data) -> np.ndarray:
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.asarray(data, dtype=float)
    if data.ndim == 4 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 3:
        raise ValueError(f"Expected (t, x, y) or (t, 1, x, y), got {data.shape}")
    return data


def to_txy(data) -> torch.Tensor:
    tensor = torch.as_tensor(data).float()
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected (t, x, y) or (t, 1, x, y), got {tuple(tensor.shape)}")
    return tensor


def normalize_01(data, normalize: bool = True) -> torch.Tensor:
    tensor = to_txy(data)
    if normalize and tensor.max() > 1.5:
        tensor = tensor / 255.0
    return tensor


def normalize_generated_dataset(images, value_min: float = 0, value_max: float = 255) -> torch.Tensor:
    data = images.float() if torch.is_tensor(images) else torch.as_tensor(images).float()
    if data.min() >= value_min and data.max() <= value_max:
        return data
    normalized = (data.clamp(-1.0, 1.0) + 1.0) * 0.5
    return normalized * (value_max - value_min) + value_min


def print_dataset_summary(name: str, images) -> None:
    tensor = ensure_tensor(images).float()
    print(
        f"{name:<28} shape={tuple(tensor.shape)!s:<18} dtype={tensor.dtype} "
        f"min={tensor.min().item():8.3f} max={tensor.max().item():8.3f} "
        f"mean={tensor.mean().item():8.3f} std={tensor.std().item():8.3f}"
    )


def print_dataset_summaries(datasets: dict[str, object]) -> None:
    for name, dataset in datasets.items():
        print_dataset_summary(name, dataset)


def show_grid(
    images,
    n: int = 16,
    cols: int = 4,
    cmap: str = "gray",
    seed: int = 0,
    save: bool = False,
    save_path: Path | str = "grid.png",
    title: str | None = None,
    colorbar: bool = False,
    vmin=None,
    vmax=None,
):
    images = squeeze_channel(images)
    rng = np.random.default_rng(seed)
    total = images.shape[0]
    n = min(int(n), int(total))
    if n <= 0:
        raise ValueError("Cannot plot an empty image batch.")

    rows = int(np.ceil(n / cols))
    indices = rng.choice(total, size=n, replace=False)

    if colorbar:
        fig = plt.figure(figsize=(cols * 2.2 + 0.45, rows * 2.2), constrained_layout=True)
        grid = fig.add_gridspec(rows, cols + 1, width_ratios=[1] * cols + [0.06], wspace=0.02)
        axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(cols)] for row in range(rows)])
        cax = fig.add_subplot(grid[:, -1])
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), squeeze=False, constrained_layout=True)
        cax = None

    im_ref = None
    for position, index in enumerate(indices):
        row, col = divmod(position, cols)
        image = images[index].detach().cpu().numpy() if torch.is_tensor(images) else np.asarray(images[index])
        im_ref = axes[row, col].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row, col].set_title(f"idx={index}")
        axes[row, col].axis("off")

    for position in range(n, rows * cols):
        row, col = divmod(position, cols)
        axes[row, col].axis("off")

    if title:
        fig.suptitle(title, fontsize=16)
    if colorbar and im_ref is not None:
        fig.colorbar(im_ref, cax=cax)
    if save:
        save_figure(fig, save_path)
    plt.show()
    return fig, indices


def save_figure(fig, path: Path | str, *, dpi: int = 300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    return path


def _read_loss_csv(csv_path: Path | str):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Loss CSV not found: {csv_path}")
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.dtype.names is None:
        raise ValueError(f"Cannot read columns from {csv_path}.")
    missing = {"epoch", "train_loss"}.difference(data.dtype.names)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}. Available: {data.dtype.names}")
    return data


def plot_loss_curve(csv_path: Path | str, save: bool = False, save_path: Path | str = "loss_curve.png"):
    data = _read_loss_csv(csv_path)
    epochs = np.atleast_1d(data["epoch"])
    train_loss = np.atleast_1d(data["train_loss"])
    val_loss = np.atleast_1d(data["val_loss"]) if "val_loss" in data.dtype.names else None
    best_loss = np.atleast_1d(data["best_loss"]) if "best_loss" in data.dtype.names else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
    axes[0].plot(epochs, train_loss, label="train_loss")
    if val_loss is not None:
        axes[0].plot(epochs, val_loss, label="val_loss", alpha=0.8)
    if best_loss is not None:
        axes[0].plot(epochs, best_loss, label="best_loss", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    eps = 1e-12
    axes[1].plot(epochs, np.log(np.clip(train_loss, eps, None)), label="log(train_loss)")
    if val_loss is not None:
        axes[1].plot(epochs, np.log(np.clip(val_loss, eps, None)), label="log(val_loss)", alpha=0.8)
    if best_loss is not None:
        axes[1].plot(epochs, np.log(np.clip(best_loss, eps, None)), label="log(best_loss)", alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Log loss")
    axes[1].set_title("Log loss")
    axes[1].legend()

    if save:
        save_figure(fig, save_path, dpi=200)
    plt.show()
    return fig


def plot_loss_curves(csv_paths: dict[object, Path], save: bool = False, save_path: Path | str = "loss_curves.png"):
    n_curves = len(csv_paths)
    n_cols = 2
    n_rows = int(np.ceil(n_curves / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), squeeze=False, constrained_layout=True)
    axes = axes.ravel()
    eps = 1e-12

    for index, (name, path) in enumerate(csv_paths.items()):
        data = _read_loss_csv(path)
        epochs = np.atleast_1d(data["epoch"])
        train_loss = np.atleast_1d(data["train_loss"])
        val_loss = np.atleast_1d(data["val_loss"]) if "val_loss" in data.dtype.names else None
        best_loss = np.atleast_1d(data["best_loss"]) if "best_loss" in data.dtype.names else None
        axis = axes[index]
        axis.plot(epochs, np.log(np.clip(train_loss, eps, None)), label="log(train_loss)")
        if val_loss is not None:
            axis.plot(epochs, np.log(np.clip(val_loss, eps, None)), label="log(val_loss)", alpha=0.8)
        if best_loss is not None:
            axis.plot(epochs, np.log(np.clip(best_loss, eps, None)), label="log(best_loss)", alpha=0.8)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Log loss")
        axis.set_title(f"Loss - {name}")
        axis.grid(alpha=0.25)
        axis.legend()

    for index in range(n_curves, len(axes)):
        axes[index].axis("off")
    if save:
        save_figure(fig, save_path, dpi=200)
    plt.show()
    return fig


def reconstruct_wavelet_sample(coeffs, wavelet: str = "db1", mode: str = "periodization") -> np.ndarray:
    coeffs = ensure_tensor(coeffs).detach().cpu().numpy()
    return pywt.idwt2((coeffs[0], (coeffs[1], coeffs[2], coeffs[3])), wavelet=wavelet, mode=mode)


def reconstruct_all_wavelet_samples(coeffs_array, wavelet: str = "db1", mode: str = "periodization") -> np.ndarray:
    coeffs_array = ensure_tensor(coeffs_array)
    return np.stack(
        [reconstruct_wavelet_sample(coeffs, wavelet=wavelet, mode=mode) for coeffs in coeffs_array],
        axis=0,
    ).astype(np.float32)


def upsample_approximation_to_image(approximation, current_level: int, wavelet: str = "db1", mode: str = "periodization"):
    approximation = ensure_tensor(approximation).detach().cpu().numpy().astype(np.float32)
    images = approximation
    for _level in range(current_level, 0, -1):
        zeros = np.zeros_like(images, dtype=np.float32)
        images = np.stack(
            [pywt.idwt2((image, (zero, zero, zero)), wavelet=wavelet, mode=mode) for image, zero in zip(images, zeros)],
            axis=0,
        )
    return images.astype(np.float32)


def reconstruct_level_projection(coeffs_array, level: int, wavelet: str = "db1", mode: str = "periodization"):
    approx = reconstruct_all_wavelet_samples(coeffs_array, wavelet=wavelet, mode=mode)
    if level == 1:
        return approx.astype(np.float32)
    return upsample_approximation_to_image(approx, current_level=level - 1, wavelet=wavelet, mode=mode)


def build_level_metric_inputs(real_coeffs, generated_coeffs, level: int, *, wavelet="db1", mode="periodization", max_images=1000):
    real_images = reconstruct_level_projection(real_coeffs[level], level, wavelet=wavelet, mode=mode)
    generated_images = reconstruct_level_projection(generated_coeffs[level], level, wavelet=wavelet, mode=mode)
    n_images = min(int(max_images), len(real_images) // 2, len(generated_images))
    if n_images == 0:
        raise ValueError(f"Not enough samples for level j{level}.")
    return real_images[:n_images], {
        "val vs val": real_images[n_images:2 * n_images],
        f"val vs generated j{level}": generated_images[:n_images],
    }, n_images


def wavelet_coeffs_to_single_channel_images(coeffs, wavelet: str = "db1", mode: str = "periodization"):
    coeffs = ensure_tensor(coeffs)
    if coeffs.ndim == 4 and coeffs.shape[1] >= 4:
        return reconstruct_all_wavelet_samples(coeffs[:, :4], wavelet=wavelet, mode=mode)
    return to_numpy_images(coeffs)


def load_baseline_data(config: BaselineConfig):
    from lib.cea_lib.signal2D import resize2d

    train_images = load_pt_dataset(config.data_root / "training.pt")
    test_images = load_pt_dataset(config.data_root / "test.pt")
    val_images = load_pt_dataset(config.data_root / "validation.pt")
    generated_dataset = normalize_generated_dataset(load_pt_dataset(config.generated_pt))
    noise_dataset = normalize_generated_dataset(load_pt_dataset(config.noise_pt))
    mnist_dataset = resize2d(load_pt_dataset(config.mnist_pt), 64, 64)
    return {
        "train": train_images,
        "test": test_images,
        "validation": val_images,
        "generated": generated_dataset,
        "noise": noise_dataset,
        "MNIST": mnist_dataset,
    }


def load_cascade_data(config: CascadeConfig):
    real_coeffs = {level: load_pt_dataset(path) for level, path in config.real_coeff_paths.items()}
    generated_coeffs = {level: load_pt_dataset(path) for level, path in config.generated_coeff_paths.items()}
    reconstructed_ca = {level: load_pt_dataset(path) for level, path in config.reconstructed_ca_paths.items()}
    final_generated_images = load_pt_dataset(config.final_generated_images_pt)
    final_real_images = reconstruct_all_wavelet_samples(real_coeffs[1], wavelet=config.wavelet, mode=config.mode)
    return {
        "real_coeffs": real_coeffs,
        "generated_coeffs": generated_coeffs,
        "reconstructed_ca": reconstructed_ca,
        "final_real_images": final_real_images,
        "final_generated_images": final_generated_images,
    }


def as_phyfid_batch(images) -> torch.Tensor:
    images = ensure_tensor(images).detach().cpu().float()
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(f"PhyFID expects scalar NCHW or NHW data, got {tuple(images.shape)}")
    return images


def compare_phyfid_stage(
    stage_name: str,
    reference_images,
    generated_images,
    encoder_path: Path | str,
    *,
    max_images: int = 1000,
    batch_size: int = 128,
    device=None,
):
    from lib.PhyFID.metrics import compare_datasets

    encoder_path = Path(encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"PhyFID encoder not found for {stage_name}: {encoder_path}")
    reference_images = as_phyfid_batch(reference_images)
    generated_images = as_phyfid_batch(generated_images)
    n_images = min(int(max_images), len(reference_images) // 2, len(generated_images))
    if n_images == 0:
        raise ValueError(f"Not enough images for PhyFID stage {stage_name}.")
    reference = reference_images[:n_images]
    reference_holdout = reference_images[n_images:2 * n_images]
    generated = generated_images[:n_images]
    return {
        "stage": stage_name,
        "n": n_images,
        "encoder": encoder_path.name,
        "ref_vs_ref": float(compare_datasets(reference, reference_holdout, encoder_path, batch_size=batch_size, device=device)),
        "ref_vs_generated": float(compare_datasets(reference, generated, encoder_path, batch_size=batch_size, device=device)),
    }


def compute_baseline_phyfid_scores(data, encoder_path, *, max_images=1000, batch_size=128, device=None):
    from lib.PhyFID.metrics import compare_datasets

    val_images = data["validation"]
    n_images = min(
        int(max_images),
        len(val_images) // 2,
        len(data["train"]),
        len(data["generated"]),
        len(data["noise"]),
        len(data["MNIST"]),
    )
    reference = val_images[:n_images]
    datasets = {
        "val vs val": val_images[n_images:2 * n_images],
        "val vs train": data["train"][:n_images],
        "val vs generated": data["generated"][:n_images],
        "val vs noise": data["noise"][:n_images],
        "val vs MNIST": data["MNIST"][:n_images],
    }
    scores = {
        name: float(compare_datasets(reference, dataset, encoder_path, batch_size=batch_size, device=device))
        for name, dataset in datasets.items()
    }
    return n_images, scores


def print_phyfid_scores(n_images: int, scores: dict[str, float]) -> None:
    print(f"PhyFID comparison on {n_images} images")
    for name, score in scores.items():
        print(f"{name:<22} {score:>10.4f}")


def compute_cascade_stage_phyfid(config: CascadeConfig, cascade_data, *, device=None):
    real_coeffs = cascade_data["real_coeffs"]
    generated_coeffs = cascade_data["generated_coeffs"]
    reconstructed_ca = cascade_data["reconstructed_ca"]
    final_real_images = cascade_data["final_real_images"]
    final_generated_images = cascade_data["final_generated_images"]

    coarse_reference = real_coeffs[config.coarse_level][:, 0:1]
    coarse_generated = generated_coeffs[config.coarse_key]
    w2_reference = upsample_approximation_to_image(
        squeeze_channel(real_coeffs[1][:, 0:1]),
        current_level=1,
        wavelet=config.wavelet,
        mode=config.mode,
    )
    w2_generated = upsample_approximation_to_image(
        squeeze_channel(reconstructed_ca[1]),
        current_level=1,
        wavelet=config.wavelet,
        mode=config.mode,
    )
    return [
        compare_phyfid_stage(
            f"coarse cA{config.coarse_level}",
            coarse_reference,
            coarse_generated,
            config.phyfid_encoder_coarse,
            max_images=config.phyfid_max_images,
            batch_size=config.phyfid_batch_size,
            device=device,
        ),
        compare_phyfid_stage(
            "w2 -> cA1 projete 64",
            w2_reference,
            w2_generated,
            config.phyfid_encoder_64,
            max_images=config.phyfid_max_images,
            batch_size=config.phyfid_batch_size,
            device=device,
        ),
        compare_phyfid_stage(
            "w1 -> image finale",
            final_real_images,
            final_generated_images,
            config.phyfid_encoder_64,
            max_images=config.phyfid_max_images,
            batch_size=config.phyfid_batch_size,
            device=device,
        ),
    ]


def print_phyfid_stage_table(rows) -> None:
    print(f"{'Etape image':<28} {'n':>6} {'encodeur':<30} {'ref vs ref':>14} {'ref vs generated':>18}")
    for row in rows:
        print(
            f"{row['stage']:<28} {row['n']:>6} {row['encoder']:<30} "
            f"{row['ref_vs_ref']:>14.4f} {row['ref_vs_generated']:>18.4f}"
        )


def prepare_images_for_fid(images) -> torch.Tensor:
    original_dtype = torch.as_tensor(images).dtype
    images = torch.as_tensor(images).detach().cpu().float()
    if images.ndim == 3:
        images = images.unsqueeze(1)
    elif images.ndim == 4 and images.shape[-1] in (1, 3) and images.shape[1] not in (1, 3):
        images = images.permute(0, 3, 1, 2)
    if images.ndim != 4:
        raise ValueError(f"Unsupported image format: {tuple(images.shape)}")
    if original_dtype == torch.uint8 or images.max() > 2:
        images = images / 255.0
    elif images.min() < 0 and images.max() <= 1:
        images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if images.shape[1] != 3:
        raise ValueError(f"FID expects 1 or 3 channels, got {images.shape[1]}")
    return images


def extract_fid_features(images, model, batch_size: int, device):
    images = prepare_images_for_fid(images)
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    features = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            prediction = model(batch.to(device))[0]
            if prediction.shape[-2:] != (1, 1):
                prediction = adaptive_avg_pool2d(prediction, output_size=(1, 1))
            features.append(prediction.flatten(1).cpu().numpy())
    return np.concatenate(features, axis=0)


def fid_from_features(reference_features, compared_features) -> float:
    from pytorch_fid.fid_score import calculate_frechet_distance

    reference_mu = np.mean(reference_features, axis=0)
    reference_sigma = np.cov(reference_features, rowvar=False)
    compared_mu = np.mean(compared_features, axis=0)
    compared_sigma = np.cov(compared_features, rowvar=False)
    return max(0.0, float(calculate_frechet_distance(reference_mu, reference_sigma, compared_mu, compared_sigma)))


def compute_inception_fid_scores(reference_images, datasets, n_images, *, dims=64, batch_size=25, device=None):
    from pytorch_fid.inception import InceptionV3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    fid_block = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    fid_model = InceptionV3([fid_block]).to(device)
    reference_features = extract_fid_features(reference_images[:n_images], fid_model, batch_size, device)
    scores = {}
    for name, dataset in datasets.items():
        compared_features = extract_fid_features(dataset[:n_images], fid_model, batch_size, device)
        scores[name] = fid_from_features(reference_features, compared_features)
    del fid_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def line_metric(validation, generated, metric_name: str = "mu"):
    validation = as_txy_numpy(validation)
    generated = as_txy_numpy(generated)
    if validation.shape[1] != generated.shape[1]:
        raise ValueError(f"Reference and generated row counts differ: {validation.shape[1]} vs {generated.shape[1]}")

    n_x = validation.shape[1]
    mu_val = validation.mean(axis=2)
    mu_gen = generated.mean(axis=2)
    if metric_name == "mu":
        ref_profile = np.mean(mu_val, axis=0)
        gen_profile = np.mean(mu_gen, axis=0)
    elif metric_name == "sigma":
        ref_profile = np.std(mu_val, axis=0)
        gen_profile = np.std(mu_gen, axis=0)
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")
    distance = np.linalg.norm(ref_profile - gen_profile) / n_x
    return float(distance), gen_profile, ref_profile


def fit_skewnorm_1d(values, min_scale: float = 1e-6, max_abs_alpha: float = 50):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0, min_scale
    mean = float(np.mean(values))
    std = float(np.std(values))
    if values.size < 3 or std < min_scale:
        return 0.0, mean, max(std, min_scale)
    try:
        alpha, loc, scale = skewnorm.fit(values)
        if not np.isfinite([alpha, loc, scale]).all() or scale < min_scale:
            raise ValueError("Invalid skew-normal parameters")
        return float(np.clip(alpha, -max_abs_alpha, max_abs_alpha)), float(loc), float(scale)
    except Exception:
        return 0.0, mean, max(std, min_scale)


def fit_skewnorm_layers(layer_values):
    layer_values = np.asarray(layer_values, dtype=float)
    params = np.array([fit_skewnorm_1d(layer_values[:, index]) for index in range(layer_values.shape[1])])
    return params[:, 0], params[:, 1], params[:, 2]


def get_alpha(data):
    data = as_txy_numpy(data)
    return fit_skewnorm_layers(data.mean(axis=2))


def skewnorm_pdf(x, alpha, loc, scale):
    scale = float(scale)
    if scale <= 0:
        return np.zeros_like(x)
    return skewnorm.pdf(x, float(alpha), loc=float(loc), scale=scale)


def plot_skewnorm_joyplot(
    n_curves,
    alpha1,
    loc1,
    scale1,
    alpha2=None,
    loc2=None,
    scale2=None,
    *,
    ax=None,
    title="",
    label1="Validation",
    label2="Generated",
    color1="gray",
    color2="tab:orange",
    x_min=-0.05,
    x_max=1.05,
):
    alpha1 = np.asarray(alpha1)
    loc1 = np.asarray(loc1)
    scale1 = np.asarray(scale1)
    if alpha1.shape != loc1.shape or loc1.shape != scale1.shape:
        raise ValueError("alpha1, loc1, and scale1 must have the same shape")

    has_second = alpha2 is not None and loc2 is not None and scale2 is not None
    if has_second:
        alpha2 = np.asarray(alpha2)
        loc2 = np.asarray(loc2)
        scale2 = np.asarray(scale2)
        if alpha2.shape != loc2.shape or loc2.shape != scale2.shape or loc2.shape != loc1.shape:
            raise ValueError("alpha2/loc2/scale2 must have the same shape as alpha1/loc1/scale1")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))

    n_curves = min(int(n_curves), len(loc1))
    selected_idx = np.linspace(0, len(loc1) - 1, n_curves, dtype=int)
    x = np.linspace(x_min, x_max, 800)
    spacing = 1.0
    height = 0.55

    for plot_idx, layer_idx in enumerate(selected_idx):
        offset = plot_idx * spacing
        y1 = skewnorm_pdf(x, alpha1[layer_idx], loc1[layer_idx], scale1[layer_idx])
        y1 = y1 / y1.max() * height if y1.max() > 0 else y1
        ax.fill_between(x, offset, offset + y1, color=color1, alpha=0.35)
        ax.plot(x, offset + y1, color=color1, linewidth=1.2, label=label1 if plot_idx == 0 else None)
        ax.axvline(
            loc1[layer_idx],
            ymin=offset / (n_curves * spacing),
            ymax=(offset + height) / (n_curves * spacing),
            color=color1,
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )
        if has_second:
            y2 = skewnorm_pdf(x, alpha2[layer_idx], loc2[layer_idx], scale2[layer_idx])
            y2 = y2 / y2.max() * height if y2.max() > 0 else y2
            ax.fill_between(x, offset, offset + y2, color=color2, alpha=0.30)
            ax.plot(x, offset + y2, color=color2, linewidth=1.2, label=label2 if plot_idx == 0 else None)
            ax.axvline(
                loc2[layer_idx],
                ymin=offset / (n_curves * spacing),
                ymax=(offset + height) / (n_curves * spacing),
                color=color2,
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
            )

    ax.set_xlim(x_min, x_max)
    ax.set_yticks(np.arange(n_curves) * spacing)
    ax.set_yticklabels([f"x={index}" for index in selected_idx])
    ax.set_xlabel(r"$\mu_l$")
    ax.set_ylabel("Row index")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    if has_second:
        ax.legend(loc="upper right")
    return ax


def row_statistic_summary(reference, datasets: dict[str, object], *, normalize_inputs: bool = True):
    reference = normalize_01(reference) if normalize_inputs else as_txy_numpy(reference)
    summary = {}
    for name, dataset in datasets.items():
        dataset = normalize_01(dataset) if normalize_inputs else as_txy_numpy(dataset)
        mu_distance, mu_profile, ref_mu_profile = line_metric(reference, dataset, metric_name="mu")
        sigma_distance, sigma_profile, ref_sigma_profile = line_metric(reference, dataset, metric_name="sigma")
        alpha, loc, scale = get_alpha(dataset)
        summary[name] = {
            "mu_distance": mu_distance,
            "sigma_distance": sigma_distance,
            "mu_profile": mu_profile,
            "sigma_profile": sigma_profile,
            "alpha": alpha,
            "loc": loc,
            "scale": scale,
            "ref_mu_profile": ref_mu_profile,
            "ref_sigma_profile": ref_sigma_profile,
        }
    alpha, loc, scale = get_alpha(reference)
    summary["reference"] = {"alpha": alpha, "loc": loc, "scale": scale}
    return summary


def _joyplot_math_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized in {"validation", "reference"}:
        return "val"
    if normalized == "noise":
        return "noise"
    if normalized == "mnist":
        return "mnist"
    return "gen"


def make_joyplot_title(
    reference_label: str,
    compared_label: str,
    metrics: dict[str, object],
    energy: bool = False,
    *,
    compared_math_label: str | None = None,
):
    prefix = "E," if energy else ""
    reference_math_label = f"{prefix}{_joyplot_math_label(reference_label)}"
    compared_math_label = f"{prefix}{compared_math_label or _joyplot_math_label(compared_label)}"
    return (
        f"{reference_label} vs {compared_label}, "
        rf"$\overline{{|\mu_{{\mathrm{{{reference_math_label}}}}} - \mu_{{\mathrm{{{compared_math_label}}}}}|^2}}$"
        f" = {metrics['mu_distance']:.4f}, "
        rf"$\overline{{|\sigma_{{\mathrm{{{reference_math_label}}}}} - \sigma_{{\mathrm{{{compared_math_label}}}}}|^2}}$"
        f" = {metrics['sigma_distance']:.4f}"
    )


def plot_density_joyplots(
    reference,
    compared: dict[str, object],
    *,
    n_curves=8,
    colors=None,
    title_labels=None,
    legend_labels=None,
    reference_color="gray",
    save_path=None,
):
    colors = colors or {"generated": "#d62728", "noise": "black", "MNIST": "black"}
    title_labels = title_labels or {}
    legend_labels = legend_labels or {}
    stats = row_statistic_summary(reference, compared, normalize_inputs=True)
    n_panels = 1 + len(compared)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    ref = stats["reference"]
    plot_skewnorm_joyplot(
        n_curves,
        ref["alpha"],
        ref["loc"],
        ref["scale"],
        ax=axes[0],
        title="Validation only",
        color1=reference_color,
    )
    for index, (name, metrics) in enumerate((item for item in stats.items() if item[0] != "reference"), start=1):
        title_label = title_labels.get(name, name)
        legend_label = legend_labels.get(name, title_label)
        plot_skewnorm_joyplot(
            n_curves,
            ref["alpha"],
            ref["loc"],
            ref["scale"],
            metrics["alpha"],
            metrics["loc"],
            metrics["scale"],
            ax=axes[index],
            title=make_joyplot_title("Validation", title_label, metrics),
            label2=legend_label,
            color1=reference_color,
            color2=colors.get(name, "tab:orange"),
        )
    for axis in axes[n_panels:]:
        axis.axis("off")
    for axis in axes[:n_panels]:
        axis.title.set_fontsize(10)
    fig.suptitle(r"Skew-normal distribution of row-averaged density $\overline{\rho_{i,x}}(y)$", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.0, w_pad=1.5)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig, stats


def transformation_fluctuation_energy_spectrum(data, axis=1):
    data = as_txy_numpy(data)
    fluctuations = np.abs(data - data.mean())
    if axis == -1:
        spectrum = np.fft.fftshift(np.fft.fft2(fluctuations, axes=(-2, -1)), axes=(-2, -1))
    else:
        spectrum = np.fft.fftshift(np.fft.fft(fluctuations, axis=axis), axes=axis)
    return np.abs(spectrum) ** 2


def plot_energy_joyplots(
    reference,
    compared: dict[str, object],
    *,
    n_curves=8,
    colors=None,
    title_labels=None,
    legend_labels=None,
    reference_color="gray",
    save_path=None,
):
    ref_energy = transformation_fluctuation_energy_spectrum(normalize_01(reference), axis=1)
    compared_energy = {
        name: transformation_fluctuation_energy_spectrum(normalize_01(dataset), axis=1)
        for name, dataset in compared.items()
    }
    colors = colors or {"generated": "#d62728", "noise": "black", "MNIST": "black"}
    title_labels = title_labels or {}
    legend_labels = legend_labels or {}
    stats = row_statistic_summary(ref_energy, compared_energy, normalize_inputs=False)
    n_panels = 1 + len(compared_energy)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    ref = stats["reference"]
    plot_skewnorm_joyplot(
        n_curves,
        ref["alpha"],
        ref["loc"],
        ref["scale"],
        ax=axes[0],
        title="Validation only",
        color1=reference_color,
        x_max=3,
    )
    for index, (name, metrics) in enumerate((item for item in stats.items() if item[0] != "reference"), start=1):
        title_label = title_labels.get(name, name)
        legend_label = legend_labels.get(name, title_label)
        plot_skewnorm_joyplot(
            n_curves,
            ref["alpha"],
            ref["loc"],
            ref["scale"],
            metrics["alpha"],
            metrics["loc"],
            metrics["scale"],
            ax=axes[index],
            title=make_joyplot_title("Validation", title_label, metrics, energy=True),
            label2=legend_label,
            color1=reference_color,
            color2=colors.get(name, "tab:orange"),
            x_max=3,
        )
    for axis in axes[n_panels:]:
        axis.axis("off")
    for axis in axes[:n_panels]:
        axis.title.set_fontsize(10)
    fig.suptitle(r"Skew-normal distribution of row-averaged fluctuation energy $\overline{E_{i,k_x}}(y)$", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.0, w_pad=1.5)
    if save_path is not None:
        save_figure(fig, save_path)
    plt.show()
    return fig, stats


def _to_unit_scale_without_clipping(images):
    data = torch.as_tensor(images).detach().cpu().float()
    if data.max() > 1.5:
        data = data / 255.0
    elif data.min() < -0.1:
        data = (data + 1.0) / 2.0
    return data


def _mean_energy_spectrum(images):
    fields = as_txy_numpy(_to_unit_scale_without_clipping(images)).astype(np.float64)
    fields = fields - fields.mean(axis=(-2, -1), keepdims=True)
    spectrum = np.fft.fftshift(np.fft.fft2(fields, axes=(-2, -1)), axes=(-2, -1))
    energy = np.abs(spectrum) ** 2
    return energy.mean(axis=0), energy.mean(axis=(-2, -1)).mean()


def spectral_energy_metrics(reference, generated, eps=1e-12):
    reference_energy, reference_total = _mean_energy_spectrum(reference)
    generated_energy, generated_total = _mean_energy_spectrum(generated)
    reference_pdf = reference_energy / max(reference_energy.sum(), eps)
    generated_pdf = generated_energy / max(generated_energy.sum(), eps)
    return {
        "spectral_l1": float(np.abs(reference_pdf - generated_pdf).sum()),
        "log_spectral_rmse": float(
            np.sqrt(np.mean((np.log10(reference_pdf + eps) - np.log10(generated_pdf + eps)) ** 2))
        ),
        "total_energy_ratio": float(generated_total / max(reference_total, eps)),
        "reference_energy": reference_energy,
        "generated_energy": generated_energy,
    }


def plot_spectral_energy(reference, generated, *, title="Fluctuation-energy spectrum"):
    metrics = spectral_energy_metrics(reference, generated)
    ref = metrics["reference_energy"]
    gen = metrics["generated_energy"]
    eps = 1e-12
    ref_log = np.log10(ref / max(ref.sum(), eps) + eps)
    gen_log = np.log10(gen / max(gen.sum(), eps) + eps)
    difference = np.abs(ref_log - gen_log)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    axes[0].imshow(ref_log, cmap="magma", origin="lower")
    axes[0].set_title("Validation")
    axes[1].imshow(gen_log, cmap="magma", origin="lower")
    axes[1].set_title("Generated")
    axes[2].imshow(difference, cmap="viridis", origin="lower")
    axes[2].set_title("|Delta log10 PSD|")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(title)
    plt.show()
    return fig, metrics


def compare_result_metrics(reference, generated, encoder_path, *, max_images=1000, batch_size=128, device="cpu"):
    from lib.PhyFID.metrics import compare_datasets

    reference = _to_unit_scale_without_clipping(reference)
    generated = _to_unit_scale_without_clipping(generated)
    n = min(int(max_images), len(reference) // 2, len(generated))
    if n == 0:
        raise ValueError("Not enough samples for a reference/generated split.")
    ref = reference[:n]
    holdout = reference[n:2 * n]
    gen = generated[:n]
    return {
        "n": n,
        "phyfid_ref_ref": float(compare_datasets(ref, holdout, encoder_path, batch_size=batch_size, device=device)),
        "phyfid_ref_generated": float(compare_datasets(ref, gen, encoder_path, batch_size=batch_size, device=device)),
        **{key: value for key, value in spectral_energy_metrics(ref, gen).items() if not key.endswith("_energy")},
    }


def print_result_metrics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"{label}: n={metrics['n']} | PhyFID(ref/ref)={metrics['phyfid_ref_ref']:.5f} | "
        f"PhyFID(ref/gen)={metrics['phyfid_ref_generated']:.5f} | "
        f"spectral L1={metrics['spectral_l1']:.5f} | "
        f"log-spectrum RMSE={metrics['log_spectral_rmse']:.5f} | "
        f"energy ratio={metrics['total_energy_ratio']:.5f}"
    )


def plot_wavelet_level_comparison(real_data, generated_data, level, n_examples=3, seed=0, channel_titles=None, cmap="gray"):
    real_data = ensure_tensor(real_data)
    generated_data = ensure_tensor(generated_data)
    n_examples = min(int(n_examples), len(real_data), len(generated_data))
    rng = np.random.default_rng(seed)
    indices = rng.choice(min(len(real_data), len(generated_data)), size=n_examples, replace=False)
    if channel_titles is None:
        channel_titles = ["Approximation cA", "Horizontal cH", "Vertical cV", "Diagonal cD"]

    fig, axes = plt.subplots(4, 2 * n_examples, figsize=(4 * n_examples, 8), squeeze=False)
    for example, index in enumerate(indices):
        real_coeff = real_data[index]
        gen_coeff = generated_data[index]
        c0 = example * 2
        for channel, channel_title in enumerate(channel_titles):
            axes[channel, c0].imshow(real_coeff[channel], cmap=cmap)
            axes[channel, c0 + 1].imshow(gen_coeff[channel], cmap=cmap)
            axes[channel, c0].set_title(f"Ref j{level}\n{channel_title}" if channel == 0 else channel_title)
            axes[channel, c0 + 1].set_title(f"Gen j{level}\n{channel_title}" if channel == 0 else channel_title)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    plt.show()
    return fig, indices


def plot_single_channel_comparison(real_images, generated_images, title, n_examples=6, seed=0, cmap="gray"):
    real_images = squeeze_channel(real_images)
    generated_images = squeeze_channel(generated_images)
    n_examples = min(int(n_examples), len(real_images), len(generated_images))
    rng = np.random.default_rng(seed)
    indices = rng.choice(min(len(real_images), len(generated_images)), size=n_examples, replace=False)
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 3 * n_examples), squeeze=False)

    for row, index in enumerate(indices):
        ref = ensure_tensor(real_images[index]).detach().cpu().numpy()
        gen = ensure_tensor(generated_images[index]).detach().cpu().numpy()
        err = np.abs(ref - gen)
        vmin = min(ref.min(), gen.min())
        vmax = max(ref.max(), gen.max())
        axes[row, 0].imshow(ref, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(gen, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row, 2].imshow(err, cmap="magma")
        axes[row, 0].set_title(f"Reference\nidx={index}")
        axes[row, 1].set_title(f"Generated\nidx={index}")
        axes[row, 2].set_title(f"|Error|\nMAE={err.mean():.3e}")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig, indices


def plot_wavelet_report_figures(
    real_coeffs,
    generated_coeffs,
    generated_images,
    indices,
    output_dir: Path | str,
    *,
    coefficient_filename="wavelet_cascade_coefficients_comparison.png",
    reconstruction_filename="wavelet_cascade_reconstruction_comparison.png",
    wavelet="db1",
    mode="periodization",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_coeffs = ensure_tensor(real_coeffs).float()
    generated_coeffs = ensure_tensor(generated_coeffs).float()
    generated_images_np = squeeze_channel(generated_images).detach().cpu().numpy()
    indices = [index for index in indices if index < min(len(real_coeffs), len(generated_coeffs), len(generated_images_np))]
    if not indices:
        raise ValueError("No valid index for report figures.")

    channel_names = [r"$c_A^1$", r"$c_H^1$", r"$c_V^1$", r"$c_D^1$"]
    fig_coeffs, axes = plt.subplots(4, 2 * len(indices), figsize=(4.5 * len(indices), 8.8), constrained_layout=True)
    if len(indices) == 1:
        axes = axes.reshape(4, 2)
    for example, index in enumerate(indices):
        c0 = 2 * example
        for channel in range(4):
            reference = real_coeffs[index, channel].detach().cpu().numpy()
            generated = generated_coeffs[index, channel].detach().cpu().numpy()
            vmax = max(abs(reference).max(), abs(generated).max())
            vmin = -vmax
            axes[channel, c0].imshow(reference, cmap="coolwarm", vmin=vmin, vmax=vmax)
            axes[channel, c0 + 1].imshow(generated, cmap="coolwarm", vmin=vmin, vmax=vmax)
            axes[channel, c0].set_ylabel(channel_names[channel], fontsize=16)
            for axis in (axes[channel, c0], axes[channel, c0 + 1]):
                axis.set_xticks([])
                axis.set_yticks([])
            if channel == 0:
                axes[channel, c0].set_title(f"Reference\nidx={index}", fontsize=16)
                axes[channel, c0 + 1].set_title(f"Cascade generated\nidx={index}", fontsize=16)
    fig_coeffs.suptitle("Final wavelet coefficient maps at level j=1", fontsize=16)
    coefficients_path = save_figure(fig_coeffs, output_dir / coefficient_filename)
    plt.show()

    reference_images = np.stack([reconstruct_wavelet_sample(real_coeffs[index], wavelet=wavelet, mode=mode) for index in indices], axis=0)
    fig_rec, axes = plt.subplots(len(indices), 3, figsize=(8.7, 2.85 * len(indices)), constrained_layout=True, squeeze=False)
    for row, index in enumerate(indices):
        reference = reference_images[row]
        generated = generated_images_np[index]
        error = np.abs(reference - generated)
        vmin = min(reference.min(), generated.min())
        vmax = max(reference.max(), generated.max())
        axes[row, 0].imshow(reference, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(generated, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 2].imshow(error, cmap="magma")
        axes[row, 0].set_ylabel(f"idx={index}", fontsize=16)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title("Reference RTI field", fontsize=16)
    axes[0, 1].set_title("Cascade generated field", fontsize=16)
    axes[0, 2].set_title("Absolute difference", fontsize=16)
    fig_rec.suptitle("RTI fields reconstructed from final cascade wavelet coefficients", fontsize=16)
    reconstruction_path = save_figure(fig_rec, output_dir / reconstruction_filename)
    plt.show()
    return coefficients_path, reconstruction_path
