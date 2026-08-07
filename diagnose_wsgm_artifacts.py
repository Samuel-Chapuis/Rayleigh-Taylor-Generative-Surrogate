"""Diagnostics rapides des artefacts périodiques et de l'énergie wavelet.

Exemple:
    python diagnose_wsgm_artifacts.py \
        outputs/generated/cascade_sgm_f/generated_images.pt \
        --real-coeffs data/RT64/processed/j1_validation.pt \
        --generated-coeffs outputs/generated/cascade_sgm_f/j1_generated_coefficients.pt

Le score ``phase_rms`` mesure la différence moyenne entre classes de phase
modulo 2 ou 4 pixels. Il ne prouve pas à lui seul un artefact, mais une hausse
à période 2/4 spécifique aux sorties générées est un signal de checkerboard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load(path: str | Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(value):
        raise TypeError(f"{path} ne contient pas un tenseur.")
    return value.float()


def phase_rms(images: torch.Tensor, period: int) -> float:
    images = images.squeeze(1) if images.ndim == 4 and images.shape[1] == 1 else images
    if images.ndim != 3:
        raise ValueError(f"Images attendues en [N,H,W], reçues {tuple(images.shape)}")
    phases = torch.stack(
        [images[:, i::period, j::period].mean(dim=(1, 2))
         for i in range(period) for j in range(period)],
        dim=1,
    )
    phases = phases - phases.mean(dim=1, keepdim=True)
    return float(phases.square().mean().sqrt())


def print_summary(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(tensor.shape)} mean={tensor.mean().item():+.6e} "
        f"std={tensor.std().item():.6e} min={tensor.min().item():+.6e} "
        f"max={tensor.max().item():+.6e}"
    )
    if tensor.ndim == 4 and tensor.shape[1] > 1:
        phase2 = [phase_rms(tensor[:, channel:channel + 1], 2)
                  for channel in range(tensor.shape[1])]
        phase4 = [phase_rms(tensor[:, channel:channel + 1], 4)
                  for channel in range(tensor.shape[1])]
        print(f"  phase_rms(p=2) per channel={phase2}")
        print(f"  phase_rms(p=4) per channel={phase4}")
    elif tensor.ndim >= 3:
        print(
            f"  phase_rms(p=2)={phase_rms(tensor, 2):.6e} "
            f"phase_rms(p=4)={phase_rms(tensor, 4):.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generated_images")
    parser.add_argument("--real-images")
    parser.add_argument("--real-coeffs")
    parser.add_argument("--generated-coeffs")
    args = parser.parse_args()

    generated_images = load(args.generated_images)
    print_summary("generated images", generated_images)
    if args.real_images:
        print_summary("real images", load(args.real_images))

    if args.real_coeffs and args.generated_coeffs:
        real = load(args.real_coeffs)
        generated = load(args.generated_coeffs)
        print_summary("real coefficients", real)
        print_summary("generated coefficients", generated)
        real_std = real[:, 1:].std(dim=(0, 2, 3))
        generated_std = generated[:, 1:].std(dim=(0, 2, 3))
        print("detail std ratio generated/real:",
              (generated_std / real_std.clamp_min(1e-12)).tolist())


if __name__ == "__main__":
    main()
