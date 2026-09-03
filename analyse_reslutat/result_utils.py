"""Utilities shared by the result-visualisation notebooks.

The functions in this module deliberately keep the oracle experiment
separate from the fully generated cascade.  In particular, replacing only
the initial cA2 by the real cA2 is useful for diagnosing error propagation,
but it is not an unconditional generation experiment.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
from scipy.stats import skewnorm

from lib.PhyFID.metrics import compare_datasets


def as_numpy_images(images):
    """Return a floating-point batch in ``(N, H, W)`` form."""
    if torch.is_tensor(images):
        images = images.detach().cpu().float().numpy()
    else:
        images = np.asarray(images, dtype=np.float32)
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    if images.ndim != 3:
        raise ValueError(f"Expected NHW or NCHW with one channel, got {images.shape}")
    return images.astype(np.float32, copy=False)


def to_unit_interval(images):
    """Convert the project's uint8, [-1, 1], or [0, 1] conventions to [0, 1]."""
    data = torch.as_tensor(images).detach().cpu().float()
    if data.max() > 1.5:
        data = data / 255.0
    elif data.min() < -0.1:
        data = (data + 1.0) / 2.0
    return data.clamp(0.0, 1.0)


def _to_unit_scale_without_clipping(images):
    """Map conventions to unit scale while preserving out-of-range artifacts."""
    data = torch.as_tensor(images).detach().cpu().float()
    if data.max() > 1.5:
        data = data / 255.0
    elif data.min() < -0.1:
        data = (data + 1.0) / 2.0
    return data


def _idwt_batch(coefficients, wavelet="db1", mode="periodization"):
    coefficients = torch.as_tensor(coefficients).detach().cpu().float()
    if coefficients.ndim != 4 or coefficients.shape[1] != 4:
        raise ValueError(f"Expected wavelet coefficients NCHW with C=4, got {tuple(coefficients.shape)}")
    reconstructed = [
        pywt.idwt2(
            (sample[0].numpy(), (sample[1].numpy(), sample[2].numpy(), sample[3].numpy())),
            wavelet=wavelet,
            mode=mode,
        )
        for sample in coefficients
    ]
    return torch.from_numpy(np.stack(reconstructed, axis=0)[:, None].astype(np.float32))


def reconstruct_oracle_cascade(
    real_coarse,
    generated_j2_coefficients,
    generated_j1_coefficients,
    wavelet="db1",
    mode="periodization",
):
    """Reconstruct a cascade with the true cA2 and generated detail bands.

    The detail bands are taken from the supplied generated cascade outputs;
    only their coarse conditioning input is replaced by the real cA2.  This
    is the controlled ``perfect coarse`` ablation used in the notebooks.
    """
    real_coarse = torch.as_tensor(real_coarse).detach().cpu().float()
    j2 = torch.as_tensor(generated_j2_coefficients).detach().cpu().float()
    j1 = torch.as_tensor(generated_j1_coefficients).detach().cpu().float()
    if real_coarse.ndim == 3:
        real_coarse = real_coarse[:, None]
    if j2.ndim != 4 or j1.ndim != 4 or real_coarse.ndim != 4:
        raise ValueError("All cascade inputs must be batched tensors.")
    if j2.shape[1] != 4 or j1.shape[1] != 4:
        raise ValueError("Generated cascade coefficients must contain [cA,cH,cV,cD].")
    if len(real_coarse) < len(j2) or len(j1) < len(j2):
        raise ValueError("Cascade inputs contain different numbers of samples.")

    c_a2 = real_coarse[:len(j2), :1]
    c_a1 = _idwt_batch(torch.cat((c_a2, j2[:, 1:]), dim=1), wavelet=wavelet, mode=mode)
    return _idwt_batch(torch.cat((c_a1, j1[:len(j2), 1:]), dim=1), wavelet=wavelet, mode=mode)


def plot_image_grid(images, *, n=16, cols=4, seed=0, cmap="gray", title=None, vmin=None, vmax=None):
    """Display a reproducible image grid; defaults to the requested 4x4 report block."""
    images = as_numpy_images(to_unit_interval(images))
    n = min(int(n), len(images))
    if n == 0:
        raise ValueError("Cannot plot an empty image batch.")
    rows = int(np.ceil(n / cols))
    indices = np.random.default_rng(seed).choice(len(images), size=n, replace=False)
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.45 * rows), squeeze=False, constrained_layout=True)
    for position, index in enumerate(indices):
        row, col = divmod(position, cols)
        axes[row, col].imshow(images[index], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row, col].set_title(f"idx={index}", fontsize=9)
        axes[row, col].axis("off")
    for position in range(n, rows * cols):
        row, col = divmod(position, cols)
        axes[row, col].axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    plt.show()
    return fig, indices


def save_figure(fig, path, *, dpi=300):
    """Save a figure after creating its parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    return path


def plot_generation_comparison(models, *, n=16, cols=4, seed=0, cmap="gray", title=None):
    """Show one reproducible 4x4 block per model in a single report figure."""
    if not models:
        raise ValueError("At least one generated dataset is required.")

    prepared = [(label, as_numpy_images(to_unit_interval(images))) for label, images in models]
    n = min(int(n), *(len(images) for _, images in prepared))
    if n == 0:
        raise ValueError("Cannot plot an empty image batch.")

    indices = np.random.default_rng(seed).choice(n, size=n, replace=False)
    rows_per_model = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows_per_model,
        len(prepared) * cols,
        figsize=(2.3 * len(prepared) * cols, 2.35 * rows_per_model),
        squeeze=False,
        constrained_layout=True,
    )
    for model_index, (label, images) in enumerate(prepared):
        for position, index in enumerate(indices):
            row, col = divmod(position, cols)
            axis = axes[row, model_index * cols + col]
            axis.imshow(images[index], cmap=cmap)
            axis.axis("off")
        for position in range(n, rows_per_model * cols):
            row, col = divmod(position, cols)
            axes[row, model_index * cols + col].axis("off")
        axes[0, model_index * cols].set_title(label, fontsize=12, loc="left", pad=7)
    if title:
        fig.suptitle(title, fontsize=14)
    return fig, indices


def _line_profiles(images):
    """Return the baseline row-mean profiles and their ensemble distances."""
    fields = as_numpy_images(_to_unit_scale_without_clipping(images)).astype(np.float64)
    return fields.mean(axis=-1)


def _line_profile_metrics_from_rows(reference_rows, generated_rows):
    if reference_rows.shape[1] != generated_rows.shape[1]:
        raise ValueError("Reference and generated fields must have the same height.")
    height = reference_rows.shape[1]
    return {
        "mean_distance": float(np.linalg.norm(reference_rows.mean(axis=0) - generated_rows.mean(axis=0)) / height),
        "std_distance": float(np.linalg.norm(reference_rows.std(axis=0) - generated_rows.std(axis=0)) / height),
    }


def line_profile_metrics(reference, generated):
    """Reproduce the baseline distances on row means and their variability."""
    return _line_profile_metrics_from_rows(_line_profiles(reference), _line_profiles(generated))


def fluctuation_energy_spectrum(images, axis=1):
    """Baseline fluctuation spectrum: ``|FFT(|rho - mean(rho)|)|^2``."""
    fields = as_numpy_images(_to_unit_scale_without_clipping(images)).astype(np.float64)
    fluctuations = np.abs(fields - fields.mean())
    spectrum = np.fft.fftshift(np.fft.fft(fluctuations, axis=axis), axes=axis)
    return np.abs(spectrum) ** 2


def _fit_skewnorm(values, min_scale=1e-6, max_abs_alpha=50):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return 0.0, float(values.mean()) if values.size else 0.0, min_scale
    mean = float(values.mean())
    std = float(values.std())
    if std < min_scale:
        return 0.0, mean, min_scale
    try:
        alpha, loc, scale = skewnorm.fit(values)
        if not np.isfinite([alpha, loc, scale]).all() or scale < min_scale:
            raise ValueError("Invalid skew-normal parameters")
        return float(np.clip(alpha, -max_abs_alpha, max_abs_alpha)), float(loc), float(scale)
    except Exception:
        return 0.0, mean, max(std, min_scale)


def plot_skewnorm_rows(reference, generated, *, quantity="density", n_curves=8, title=None):
    """Compare row-wise skew-normal laws on a deliberately small diagnostic batch.

    Only the displayed rows are fitted.  This retains the baseline diagnostic
    while avoiding a fit for every row of every image ensemble.
    """
    if quantity == "density":
        reference_data = as_numpy_images(_to_unit_scale_without_clipping(reference))
        generated_data = as_numpy_images(_to_unit_scale_without_clipping(generated))
        x_label = "Row-averaged density"
        metric_label = "density"
    elif quantity == "energy":
        reference_data = fluctuation_energy_spectrum(reference)
        generated_data = fluctuation_energy_spectrum(generated)
        x_label = "Row-averaged fluctuation energy"
        metric_label = "energy"
    else:
        raise ValueError("quantity must be either 'density' or 'energy'.")

    reference_rows = reference_data.mean(axis=-1)
    generated_rows = generated_data.mean(axis=-1)
    if reference_rows.shape[1] != generated_rows.shape[1]:
        raise ValueError("Reference and generated fields must have the same height.")

    if quantity == "density":
        metrics = line_profile_metrics(reference, generated)
    else:
        metrics = _line_profile_metrics_from_rows(reference_rows, generated_rows)

    indices = np.linspace(0, reference_rows.shape[1] - 1, min(int(n_curves), reference_rows.shape[1]), dtype=int)
    values = np.concatenate((reference_rows[:, indices].ravel(), generated_rows[:, indices].ravel()))
    x_min, x_max = np.quantile(values[np.isfinite(values)], [0.005, 0.995])
    if x_max <= x_min:
        x_max = x_min + 1e-6
    x = np.linspace(x_min, x_max, 500)

    fig, axis = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)
    spacing, height = 1.0, 0.64
    for offset_index, row_index in enumerate(indices):
        offset = offset_index * spacing
        for values_at_row, color, label in (
            (reference_rows[:, row_index], "#1f77b4", "Validation"),
            (generated_rows[:, row_index], "#d62728", "Generated"),
        ):
            alpha, loc, scale = _fit_skewnorm(values_at_row)
            curve = skewnorm.pdf(x, alpha, loc=loc, scale=scale)
            if curve.max() > 0:
                curve = height * curve / curve.max()
            axis.plot(x, offset + curve, color=color, linewidth=1.4, label=label if offset_index == 0 else None)
            axis.fill_between(x, offset, offset + curve, color=color, alpha=0.18)
    axis.set_yticks(np.arange(len(indices)) * spacing, [str(index) for index in indices])
    axis.set_ylabel("Row index")
    axis.set_xlabel(x_label)
    axis.grid(alpha=0.2, axis="x")
    axis.legend(loc="upper right")
    axis.set_title(
        title or (
            f"Row-wise skew-normal comparison ({metric_label}): "
            f"d_mean={metrics['mean_distance']:.3e}, d_std={metrics['std_distance']:.3e}"
        )
    )
    return fig, metrics


def _mean_energy_spectrum(images):
    fields = as_numpy_images(_to_unit_scale_without_clipping(images)).astype(np.float64)
    fields = fields - fields.mean(axis=(-2, -1), keepdims=True)
    spectrum = np.fft.fftshift(np.fft.fft2(fields, axes=(-2, -1)), axes=(-2, -1))
    energy = np.abs(spectrum) ** 2
    return energy.mean(axis=0), energy.mean(axis=(-2, -1)).mean()


def spectral_energy_metrics(reference, generated, eps=1e-12):
    """Compare normalized 2-D fluctuation-energy spectra.

    ``spectral_l1`` compares spectral shape after unit-integral
    normalization. ``log_spectral_rmse`` emphasizes discrepancies over
    several scales. ``total_energy_ratio`` remains sensitive to a global
    loss or excess of fluctuation energy.
    """
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
    """Plot reference, generated, and absolute log-spectrum difference."""
    metrics = spectral_energy_metrics(reference, generated)
    ref = metrics["reference_energy"]
    gen = metrics["generated_energy"]
    eps = 1e-12
    ref_log = np.log10(ref / max(ref.sum(), eps) + eps)
    gen_log = np.log10(gen / max(gen.sum(), eps) + eps)
    difference = np.abs(ref_log - gen_log)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    common = {"cmap": "magma", "origin": "lower"}
    axes[0].imshow(ref_log, **common)
    axes[0].set_title("Validation")
    axes[1].imshow(gen_log, **common)
    axes[1].set_title("Généré")
    axes[2].imshow(difference, cmap="viridis", origin="lower")
    axes[2].set_title("|Δ log10 PSD|")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(title)
    plt.show()
    return fig, metrics


def compare_result_metrics(reference, generated, encoder_path, *, max_images=1000, batch_size=128):
    """Return PhyFID and spectral metrics using a common validation split."""
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
        "phyfid_ref_ref": float(compare_datasets(ref, holdout, encoder_path, batch_size=batch_size, device="cpu")),
        "phyfid_ref_generated": float(compare_datasets(ref, gen, encoder_path, batch_size=batch_size, device="cpu")),
        **{key: value for key, value in spectral_energy_metrics(ref, gen).items() if not key.endswith("_energy")},
    }


def print_result_metrics(label, metrics):
    print(
        f"{label}: n={metrics['n']} | PhyFID(ref/ref)={metrics['phyfid_ref_ref']:.5f} | "
        f"PhyFID(ref/gen)={metrics['phyfid_ref_generated']:.5f} | "
        f"spectral L1={metrics['spectral_l1']:.5f} | "
        f"log-spectrum RMSE={metrics['log_spectral_rmse']:.5f} | "
        f"energy ratio={metrics['total_energy_ratio']:.5f}"
    )
