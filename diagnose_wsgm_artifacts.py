"""Diagnostics rapides des artefacts périodiques et de l'énergie wavelet.

Exemple:
    python diagnose_wsgm_artifacts.py \
        outputs/generated/cascade_sgm_f/generated_images.pt \
        --real-coeffs data/RT64/processed/j1_validation.pt \
        --generated-coeffs outputs/generated/cascade_sgm_f/j1_generated_coefficients.pt

Le score ``phase_rms`` mesure la différence moyenne entre classes de phase
modulo 2 ou 4 pixels. Il ne prouve pas à lui seul un artefact, mais une hausse
à période 2/4 spécifique aux sorties générées est un signal de checkerboard.

Mode cascade:
    python diagnose_wsgm_artifacts.py \
        outputs/generated/cascade_sgm_hcirc_blurpool/generated_images.pt \
        --gen-dir outputs/generated/cascade_sgm_hcirc_blurpool \
        --real-root data/RT64/processed \
        --split validation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


PERIODS = (2, 4)


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


def phase_rms_channels(tensor: torch.Tensor, period: int) -> list[float]:
    if tensor.ndim != 4:
        return [phase_rms(tensor, period)]
    return [phase_rms(tensor[:, channel:channel + 1], period)
            for channel in range(tensor.shape[1])]


def roughness(tensor: torch.Tensor) -> float:
    """Energie moyenne des gradients finis, utile pour quantifier le bruit pixel."""
    if tensor.ndim == 3:
        tensor = tensor[:, None]
    if tensor.ndim != 4:
        raise ValueError(f"Tenseur attendu en [N,C,H,W] ou [N,H,W], recu {tuple(tensor.shape)}")
    dy = tensor[..., 1:, :] - tensor[..., :-1, :]
    dx = tensor[..., :, 1:] - tensor[..., :, :-1]
    return float(0.5 * (dy.square().mean() + dx.square().mean()))


def checkerboard_ratio(tensor: torch.Tensor, period: int) -> list[float]:
    """phase_rms/std par canal: plus stable qu'un phase_rms absolu."""
    if tensor.ndim == 3:
        tensor = tensor[:, None]
    std = tensor.std(dim=(0, 2, 3)).clamp_min(1e-12)
    return [value / float(std[index]) for index, value in enumerate(phase_rms_channels(tensor, period))]


def print_summary(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(tensor.shape)} mean={tensor.mean().item():+.6e} "
        f"std={tensor.std().item():.6e} min={tensor.min().item():+.6e} "
        f"max={tensor.max().item():+.6e}"
    )
    if tensor.ndim >= 3:
        print(f"  roughness={roughness(tensor):.6e}")
    if tensor.ndim == 4 and tensor.shape[1] > 1:
        for period in PERIODS:
            print(f"  phase_rms(p={period}) per channel={phase_rms_channels(tensor, period)}")
            print(f"  phase/std(p={period}) per channel={checkerboard_ratio(tensor, period)}")
    elif tensor.ndim >= 3:
        for period in PERIODS:
            print(
                f"  phase_rms(p={period})={phase_rms(tensor, period):.6e} "
                f"phase/std(p={period})={checkerboard_ratio(tensor, period)[0]:.6e}"
            )


def compare(name: str, real: torch.Tensor, generated: torch.Tensor) -> None:
    n = min(len(real), len(generated))
    real = real[:n]
    generated = generated[:n]
    print(f"\n[{name}] n={n}")
    print_summary("  real", real)
    print_summary("  generated", generated)
    if real.ndim == 4 and generated.ndim == 4 and real.shape[1] == generated.shape[1]:
        real_std = real.std(dim=(0, 2, 3)).clamp_min(1e-12)
        gen_std = generated.std(dim=(0, 2, 3))
        real_abs = real.abs().mean(dim=(0, 2, 3)).clamp_min(1e-12)
        gen_abs = generated.abs().mean(dim=(0, 2, 3))
        print("  std ratio generated/real:", (gen_std / real_std).tolist())
        print("  mean|x| ratio generated/real:", (gen_abs / real_abs).tolist())


def maybe_load(path: Path) -> torch.Tensor | None:
    if not path.exists():
        print(f"[missing] {path}")
        return None
    return load(path)


def run_cascade_diagnostics(gen_dir: Path, real_root: Path, split: str) -> None:
    real_j1 = maybe_load(real_root / f"j1_{split}.pt")
    real_j2 = maybe_load(real_root / f"j2_{split}.pt")
    gen_j1 = maybe_load(gen_dir / "j1_generated_coefficients.pt")
    gen_j2 = maybe_load(gen_dir / "j2_generated_coefficients.pt")
    rec_ca1 = maybe_load(gen_dir / "j1_reconstructed_cA.pt")
    gen_images = maybe_load(gen_dir / "generated_images.pt")

    if real_j2 is not None and gen_j2 is not None:
        compare("j2 coefficients [cA2,cH2,cV2,cD2]", real_j2, gen_j2)
        compare("j2 details only", real_j2[:, 1:], gen_j2[:, 1:])

    if real_j1 is not None and rec_ca1 is not None:
        compare("cA1 reconstructed from generated j2 vs real cA1", real_j1[:, :1], rec_ca1)

    if real_j1 is not None and gen_j1 is not None:
        compare("j1 coefficients [cA1,cH1,cV1,cD1]", real_j1, gen_j1)
        compare("j1 details only", real_j1[:, 1:], gen_j1[:, 1:])

    if gen_images is not None:
        print("\n[final generated images]")
        print_summary("generated images", gen_images)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generated_images")
    parser.add_argument("--real-images")
    parser.add_argument("--real-coeffs")
    parser.add_argument("--generated-coeffs")
    parser.add_argument("--gen-dir", help="Dossier de sortie complet de generate_cascade.")
    parser.add_argument("--real-root", default="data/RT64/processed")
    parser.add_argument("--split", default="validation")
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

    if args.gen_dir:
        run_cascade_diagnostics(Path(args.gen_dir), Path(args.real_root), args.split)


if __name__ == "__main__":
    main()
