from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "doc" / "poster" / "figures"
DATA_ROOT = PROJECT_ROOT / "data" / "RT64" / "processed"
CASCADE_ROOT = PROJECT_ROOT / "outputs" / "generated" / "cascade"


def as_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True).float()


def squeeze_images(images):
    if images.ndim == 4 and images.shape[1] == 1:
        return images[:, 0]
    return images


def reconstruct_wavelet(coeffs, wavelet="db1", mode="periodization"):
    coeffs = coeffs.detach().cpu().numpy()
    images = []
    for sample in coeffs:
        c_a, c_h, c_v, c_d = sample[:4]
        images.append(pywt.idwt2((c_a, (c_h, c_v, c_d)), wavelet=wavelet, mode=mode))
    return torch.as_tensor(np.stack(images, axis=0)).float()


def robust_limits(*arrays):
    values = np.concatenate([np.ravel(np.asarray(a)) for a in arrays])
    return np.quantile(values, [0.01, 0.99])


def save_reconstruction_comparison(real, generated):
    real = squeeze_images(real)
    generated = squeeze_images(generated)
    indices = [2, 11, 27]
    vmin, vmax = robust_limits(real[indices], generated[indices])

    fig, axes = plt.subplots(
        len(indices),
        3,
        figsize=(8.4, 7.2),
        constrained_layout=True,
    )
    for row, idx in enumerate(indices):
        dns = real[idx].numpy()
        gen = generated[idx].numpy()
        err = np.abs(dns - gen)

        axes[row, 0].imshow(dns, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(gen, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row, 2].imshow(err, cmap="magma", vmin=0, vmax=np.quantile(err, 0.99))

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    for ax, title in zip(axes[0], ["DNS reference", "Wavelet cascade", "Absolute error"]):
        ax.set_title(title, fontsize=13, weight="bold")

    fig.savefig(FIGURE_DIR / "cascade_reconstruction_comparison.png", dpi=300)
    plt.close(fig)


def save_generated_grid(generated):
    generated = squeeze_images(generated)
    indices = [0, 5, 9, 14, 21, 25, 32, 41]
    vmin, vmax = robust_limits(generated[indices])

    fig, axes = plt.subplots(2, 4, figsize=(8.8, 4.4), constrained_layout=True)
    for ax, idx in zip(axes.ravel(), indices):
        ax.imshow(generated[idx].numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(FIGURE_DIR / "cascade_generated_samples.png", dpi=300)
    plt.close(fig)


def save_metric_summary():
    labels = ["Noise", "Direct DDPM", "Wavelet cascade"]
    phyfid = [6.04, 0.132, 3.262]
    colors = ["#8c8c8c", "#0072B2", "#E50019"]

    fig, ax = plt.subplots(figsize=(6.2, 3.2), constrained_layout=True)
    bars = ax.bar(labels, phyfid, color=colors)
    ax.set_ylabel("PhyFID lower is better")
    ax.set_ylim(0, 6.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, phyfid):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.12,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
    fig.savefig(FIGURE_DIR / "phyfid_summary.png", dpi=300)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    real_coeffs_j1 = as_tensor(DATA_ROOT / "j1_validation.pt")
    generated = as_tensor(CASCADE_ROOT / "generated_images.pt")
    real = reconstruct_wavelet(real_coeffs_j1)

    save_reconstruction_comparison(real, generated)
    save_generated_grid(generated)
    save_metric_summary()


if __name__ == "__main__":
    main()
