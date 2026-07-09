"""Conditional 1D diffusion model on RT-CEA wavelet profiles.

The script builds one-dimensional profiles from preprocessed RT-CEA images,
splits each profile into level-1 wavelet coefficients, and trains a small MLP
to denoise the high-frequency coefficients ``Ch`` conditionally on the
low-frequency coefficients ``Ca``.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTORCH_FID_SRC = REPO_ROOT / "lib" / "pytorch-fid-master" / "src"
if str(PYTORCH_FID_SRC) not in sys.path:
    sys.path.insert(0, str(PYTORCH_FID_SRC))

from lib.cea_lib.data_loader import data_preprocessing, load_RTCEA


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the conditional wavelet diffusion experiment."""

    # Data.
    requested_signal_count: int = 50_000
    signal_size: int = 32
    data_file: str = "data/RTCEA_bimode.hdf5"

    # Wavelet transform.
    wavelet: str = "db1"
    diagnostic_levels: tuple[int, ...] = (2, 3)

    # Forward/reverse diffusion.
    beta: float = 0.02
    snr_threshold: float = 3.0

    # Model and optimization.
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 256
    epochs_list: tuple[int, ...] = (100, 200)

    # Evaluation/output.
    generated_count: int = 5_000
    seed: int = 42
    output_dir: Path = Path("hyperparameter_results")


def get_device() -> torch.device:
    """Return the accelerator used for model training."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch random generators."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_rtcea_images(data_file: str, signal_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load RT-CEA images and resize/crop them to square arrays.

    If OpenCV is unavailable in the local CEA preprocessing utilities, the
    function falls back to PyTorch bilinear interpolation after crop/mask.
    """

    data_raw, labels_raw = load_RTCEA(data_file)

    try:
        data_norm, metadata_labels = data_preprocessing(
            data_raw,
            labels_raw,
            resize=signal_size,
        )
    except ImportError as exc:
        if "OpenCV" not in str(exc):
            raise

        data_cropped, metadata_labels = data_preprocessing(
            data_raw,
            labels_raw,
            resize=-1,
        )
        data_tensor = torch.as_tensor(data_cropped, dtype=torch.float32)[:, None]
        data_norm = F.interpolate(
            data_tensor,
            size=(signal_size, signal_size),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()

    if data_norm.shape[0] == 0:
        raise ValueError(
            "data_preprocessing n'a retourné aucune image. "
            "Vérifier le fichier et les critères de crop."
        )

    return data_raw, data_norm, metadata_labels


def build_normalized_profiles(
    images: np.ndarray,
    requested_signal_count: int,
    signal_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten images into 1D profiles and normalize them globally.

    Each image of shape ``(signal_size, signal_size)`` contributes
    ``signal_size`` horizontal profiles. The final tensor is normalized with a
    single dataset-wide mean and standard deviation, which is consistent with a
    Gaussian diffusion prior.
    """

    image_count = requested_signal_count // signal_size
    profile_images = images[:image_count]
    profiles = profile_images.reshape(-1, signal_size)[:requested_signal_count]

    dataset = torch.as_tensor(profiles, dtype=torch.float32)
    signal_mean = dataset.mean()
    signal_std = dataset.std().clamp_min(1e-8)
    dataset = (dataset - signal_mean) / signal_std

    return dataset, signal_mean, signal_std


def wavelet_transform_1d(
    signals: torch.Tensor | np.ndarray,
    wavelet: str = "db1",
    level: int = 1,
) -> torch.Tensor | list[list[np.ndarray]]:
    """Compute a 1D wavelet decomposition for each signal.

    For ``level == 1``, returns a tensor of shape ``(N, 2, L/2)`` where channel
    0 is ``cA1`` and channel 1 is ``cD1``. For deeper levels, PyWavelets returns
    coefficient blocks of unequal sizes, so the natural list representation is
    preserved.
    """

    if isinstance(signals, torch.Tensor):
        signals_np = signals.detach().cpu().numpy()
        output_dtype = signals.dtype
    else:
        signals_np = np.asarray(signals)
        output_dtype = torch.float32

    coefficients = [
        pywt.wavedec(signal, wavelet=wavelet, level=level)
        for signal in signals_np
    ]

    if level != 1:
        return coefficients

    stacked = np.stack(
        [np.stack(sample_coeffs, axis=0) for sample_coeffs in coefficients],
        axis=0,
    )
    return torch.as_tensor(stacked, dtype=output_dtype)


def split_level1_coefficients(j1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split level-1 coefficients into low-pass ``Ca`` and high-pass ``Ch``."""

    if j1.ndim != 3 or j1.shape[1] != 2:
        raise ValueError(f"Expected j1 with shape (N, 2, L/2), got {tuple(j1.shape)}.")
    return j1[:, 0, :], j1[:, 1, :]


def inverse_wavelet_transform_1d(
    j1: torch.Tensor | np.ndarray,
    wavelet: str = "db1",
) -> torch.Tensor:
    """Reconstruct signals from level-1 coefficients of shape ``(N, 2, L/2)``."""

    if isinstance(j1, torch.Tensor):
        coefficients_np = j1.detach().cpu().numpy()
        output_dtype = j1.dtype
    else:
        coefficients_np = np.asarray(j1)
        output_dtype = torch.float32

    reconstructed = [
        pywt.waverec([sample[0], sample[1]], wavelet=wavelet)
        for sample in coefficients_np
    ]
    return torch.as_tensor(np.stack(reconstructed, axis=0), dtype=output_dtype)


def rmse(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    """Root-mean-square error between two tensors."""

    return torch.mean((estimate - reference) ** 2).sqrt()


def compute_time_steps_from_snr(beta: float, snr_threshold: float) -> int:
    """Smallest diffusion step count such that alpha_bar/(1-alpha_bar) < threshold."""

    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}.")
    if snr_threshold <= 0.0:
        raise ValueError(f"snr_threshold must be positive, got {snr_threshold}.")

    alpha = 1.0 - beta
    alpha_bar = 1.0
    time_steps = 0

    while True:
        time_steps += 1
        alpha_bar *= alpha
        snr = alpha_bar / (1.0 - alpha_bar)
        if snr < snr_threshold:
            return time_steps


def diffuse_fixed_steps(
    data: torch.Tensor,
    steps: int = 100,
    beta: float = 0.02,
) -> tuple[list[torch.distributions.Normal | None], list[torch.Tensor]]:
    """Apply the Markov forward diffusion process for a fixed number of steps."""

    distributions: list[torch.distributions.Normal | None] = [None]
    samples = [data]
    xt = data

    for _ in range(steps):
        distribution = torch.distributions.Normal(
            np.sqrt(1.0 - beta) * xt,
            np.sqrt(beta),
        )
        xt = distribution.sample()
        distributions.append(distribution)
        samples.append(xt)

    return distributions, samples


def diffuse_until_snr(
    data: torch.Tensor,
    snr_threshold: float = 1e-1,
    beta: float = 0.02,
) -> tuple[list[torch.distributions.Normal | None], list[torch.Tensor], int]:
    """Apply forward diffusion until the theoretical SNR is below a threshold."""

    time_steps = compute_time_steps_from_snr(beta=beta, snr_threshold=snr_threshold)
    distributions, samples = diffuse_fixed_steps(data, steps=time_steps, beta=beta)
    return distributions, samples, time_steps


def create_model(
    target_dim: int,
    condition_dim: int,
    hidden_dim: int,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Create the epsilon predictor and its AdamW optimizer."""

    input_dim = target_dim + 1 + condition_dim
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, target_dim),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    return model, optimizer


def diffusion_loss(
    model: torch.nn.Module,
    ch0: torch.Tensor,
    ca_condition: torch.Tensor,
    time_steps: int,
    beta: float,
) -> torch.Tensor:
    """DDPM epsilon-prediction loss for conditional high-frequency coefficients."""

    batch_size = ch0.shape[0]
    t = torch.randint(
        1,
        time_steps + 1,
        (batch_size, 1),
        device=ch0.device,
        dtype=ch0.dtype,
    )

    alpha_bar = (1.0 - beta) ** t
    eps = torch.randn_like(ch0)
    cht = torch.sqrt(alpha_bar) * ch0 + torch.sqrt(1.0 - alpha_bar) * eps

    model_input = torch.cat(
        [
            cht,
            t / time_steps,
            ca_condition.to(device=ch0.device, dtype=ch0.dtype),
        ],
        dim=1,
    )
    eps_pred = model(model_input)

    return F.mse_loss(eps_pred, eps)


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ch: torch.Tensor,
    ca: torch.Tensor,
    epochs: int,
    time_steps: int,
    beta: float,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    """Train the conditional epsilon predictor and return per-batch losses."""

    train_loader = DataLoader(
        TensorDataset(ch, ca),
        batch_size=batch_size,
        shuffle=True,
    )
    loss_history: list[float] = []

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for ch_batch, ca_batch in pbar:
            ch_batch = ch_batch.to(device)
            ca_batch = ca_batch.to(device)

            optimizer.zero_grad()
            loss = diffusion_loss(
                model=model,
                ch0=ch_batch,
                ca_condition=ca_batch,
                time_steps=time_steps,
                beta=beta,
            )
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            loss_history.append(loss_value)
            pbar.set_postfix(loss=f"{loss_value:.4f}")

    return loss_history


@torch.no_grad()
def sample_reverse(
    model: torch.nn.Module,
    ca_condition: torch.Tensor,
    steps: int,
    target_dim: int,
    beta: float,
) -> torch.Tensor:
    """Sample high-frequency coefficients with the reverse DDPM recursion."""

    count = ca_condition.shape[0]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    ca_condition = ca_condition.to(device=device, dtype=dtype)
    xt = torch.randn(count, target_dim, device=device, dtype=dtype)
    sample_history = [xt.detach().cpu()]

    alpha = 1.0 - beta
    for t in range(steps, 0, -1):
        t_feature = torch.full(
            (count, 1),
            t / steps,
            device=device,
            dtype=dtype,
        )
        model_input = torch.cat([xt, t_feature, ca_condition], dim=1)
        eps_pred = model(model_input)

        alpha_bar = alpha**t
        mean = (1.0 / np.sqrt(alpha)) * (
            xt - (beta / np.sqrt(1.0 - alpha_bar)) * eps_pred
        )

        if t > 1:
            xt = mean + np.sqrt(beta) * torch.randn_like(xt)
        else:
            xt = mean

        sample_history.append(xt.detach().cpu())

    return torch.stack(sample_history)


def prepare_wavelet_dataset(
    dataset: torch.Tensor,
    wavelet: str,
    diagnostic_levels: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute wavelet coefficients and print reconstruction diagnostics."""

    j1 = wavelet_transform_1d(dataset, wavelet=wavelet, level=1)
    if not isinstance(j1, torch.Tensor):
        raise TypeError("Level-1 wavelet transform should return a tensor.")

    ca, ch = split_level1_coefficients(j1)
    reconstructed = inverse_wavelet_transform_1d(j1, wavelet=wavelet)[:, : dataset.shape[1]]
    reconstruction_error = rmse(dataset.cpu(), reconstructed)

    print("dataset.shape =", dataset.shape)
    print("j1.shape      =", j1.shape, "  # (N, 2, SIGNAL_SIZE/2)")
    print("Ca.shape      =", ca.shape, "  # condition basse fréquence")
    print("Ch.shape      =", ch.shape, "  # variable diffusée haute fréquence")

    for level in diagnostic_levels:
        coeffs = wavelet_transform_1d(dataset, wavelet=wavelet, level=level)
        if isinstance(coeffs, list) and coeffs:
            print(f"j{level}[0] shapes  =", [coef.shape for coef in coeffs[0]])

    print("RMSE reconstruction niveau 1 =", reconstruction_error.item())
    return j1, ca, ch


def evaluate_generated_signals(
    generated_ch: torch.Tensor,
    ca_condition: torch.Tensor,
    original_signals: torch.Tensor,
    wavelet: str,
    signal_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct generated profiles and compare them with original profiles."""

    generated_j1 = torch.stack([ca_condition.cpu(), generated_ch.cpu()], dim=1)
    generated_signals = inverse_wavelet_transform_1d(generated_j1, wavelet=wavelet)[
        :, :signal_size
    ]
    generated_rmse = rmse(original_signals.cpu(), generated_signals)
    return generated_signals, generated_rmse


def save_rmse_csv(
    generated_rmse: dict[tuple[int, int], torch.Tensor],
    output_dir: Path,
) -> Path:
    """Write generated-signal RMSE values to a CSV file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rmse_csv_path = output_dir / "generated_rmse_hyperparameters.csv"

    with rmse_csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["EPOCHS_NB", "TIME_STEPS", "RMSE"])
        for (epochs, time_steps), value in sorted(generated_rmse.items()):
            writer.writerow([epochs, time_steps, value.item()])

    return rmse_csv_path


def plot_loss_histories(
    loss_histories: dict[tuple[int, int], list[float]],
    output_dir: Path,
) -> Path:
    """Save the training-loss curves for all hyperparameter runs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "loss_history_hyperparameters.png"

    plt.figure(figsize=(12, 6))
    for (epochs, time_steps), loss_history in sorted(loss_histories.items()):
        plt.plot(loss_history, label=f"E={epochs}, T={time_steps}", alpha=0.5)

    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training loss for all hyperparameter combinations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

    return figure_path


def plot_rmse_heatmap(
    generated_rmse: dict[tuple[int, int], torch.Tensor],
    epochs_list: tuple[int, ...],
    time_steps_list: tuple[int, ...],
    output_dir: Path,
) -> Path:
    """Save a heatmap of RMSE values indexed by epoch count and time steps."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "generated_rmse_heatmap.png"

    rmse_matrix = np.full((len(time_steps_list), len(epochs_list)), np.nan)
    for i, time_steps in enumerate(time_steps_list):
        for j, epochs in enumerate(epochs_list):
            rmse_matrix[i, j] = generated_rmse[(epochs, time_steps)].item()

    plt.figure(figsize=(8, 6))
    image = plt.imshow(rmse_matrix, origin="lower", aspect="auto")
    plt.colorbar(image, label="RMSE")
    plt.xticks(np.arange(len(epochs_list)), epochs_list)
    plt.yticks(np.arange(len(time_steps_list)), time_steps_list)
    plt.xlabel("EPOCHS_NB")
    plt.ylabel("TIME_STEPS")
    plt.title("RMSE heatmap")

    for i in range(len(time_steps_list)):
        for j in range(len(epochs_list)):
            plt.text(
                j,
                i,
                f"{rmse_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white",
            )

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

    return figure_path


def print_data_summary(
    data_raw: np.ndarray,
    data_norm: np.ndarray,
    dataset: torch.Tensor,
    config: ExperimentConfig,
) -> None:
    """Print the main shapes and normalization diagnostics."""

    image_count = config.requested_signal_count // config.signal_size
    print("data_raw.shape       =", data_raw.shape)
    print("data_norm.shape      =", data_norm.shape)
    print("profile_images.shape =", data_norm[:image_count].shape)
    print("dataset.shape        =", dataset.shape)
    print("mean/std             =", dataset.mean().item(), dataset.std().item())


def run_experiment(config: ExperimentConfig) -> None:
    """Run the full conditional wavelet diffusion experiment."""

    device = get_device()
    data_raw, data_norm, _ = preprocess_rtcea_images(
        data_file=config.data_file,
        signal_size=config.signal_size,
    )
    dataset, _, _ = build_normalized_profiles(
        images=data_norm,
        requested_signal_count=config.requested_signal_count,
        signal_size=config.signal_size,
    )
    print_data_summary(data_raw, data_norm, dataset, config)

    _, ca, ch = prepare_wavelet_dataset(
        dataset=dataset,
        wavelet=config.wavelet,
        diagnostic_levels=config.diagnostic_levels,
    )

    target_dim = ch.shape[1]
    condition_dim = ca.shape[1]
    generated_count = min(config.generated_count, dataset.shape[0])
    ca_condition = ca[:generated_count]
    original_signals = dataset[:generated_count]

    real_j1 = torch.stack([ca_condition.cpu(), ch[:generated_count].cpu()], dim=1)
    exact_reconstruction = inverse_wavelet_transform_1d(real_j1, wavelet=config.wavelet)[
        :, : config.signal_size
    ]
    exact_rmse = rmse(original_signals.cpu(), exact_reconstruction)
    print("RMSE reconstruction exacte des originaux =", exact_rmse.item())

    time_steps = compute_time_steps_from_snr(
        beta=config.beta,
        snr_threshold=config.snr_threshold,
    )
    time_steps_list = (time_steps,)
    print(f"Diffusion time steps from SNR threshold = {time_steps}")

    loss_histories: dict[tuple[int, int], list[float]] = {}
    generated_ch: dict[tuple[int, int], torch.Tensor] = {}
    generated_rmse: dict[tuple[int, int], torch.Tensor] = {}

    for epochs in config.epochs_list:
        set_seed(config.seed)
        model, optimizer = create_model(
            target_dim=target_dim,
            condition_dim=condition_dim,
            hidden_dim=config.hidden_dim,
            device=device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        print(f"Training with EPOCHS_NB={epochs}, TIME_STEPS={time_steps}")
        key = (epochs, time_steps)
        loss_histories[key] = train_model(
            model=model,
            optimizer=optimizer,
            ch=ch,
            ca=ca,
            epochs=epochs,
            time_steps=time_steps,
            beta=config.beta,
            batch_size=config.batch_size,
            device=device,
        )

        samples_ch = sample_reverse(
            model=model,
            ca_condition=ca_condition,
            steps=time_steps,
            target_dim=target_dim,
            beta=config.beta,
        )
        generated_ch[key] = samples_ch

        _, generated_rmse[key] = evaluate_generated_signals(
            generated_ch=samples_ch[-1],
            ca_condition=ca_condition,
            original_signals=original_signals,
            wavelet=config.wavelet,
            signal_size=config.signal_size,
        )
        print(f"Generated RMSE for E={epochs}, T={time_steps}: {generated_rmse[key].item()}")

    output_dir = config.output_dir
    save_rmse_csv(generated_rmse, output_dir)
    plot_loss_histories(loss_histories, output_dir)
    plot_rmse_heatmap(generated_rmse, config.epochs_list, time_steps_list, output_dir)


def main() -> None:
    """Script entry point."""

    run_experiment(ExperimentConfig())


if __name__ == "__main__":
    main()
