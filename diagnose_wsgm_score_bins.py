"""Evaluate a trained WSGM level by diffusion-time bins.

This script does not train. It loads one saved level config/checkpoint and
measures whether the score model is specifically weak near t ~= 0, where
residual visual noise is decided.

Example:
    python diagnose_wsgm_score_bins.py \
        outputs/model/cascade_sgm_hcirc_blurpool/wave_j1_RT64_hcirc_blurpool_config.json \
        --max-batches 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from WSGM_Foward_and_Generator import absolute_path, build_sgm
from lib.diffusion_lib.utils import get_best_device
from lib.wavelet_diffusion_lib.wavelet_utils import load_wave_tensor, normalize_with_stats


DEFAULT_BINS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def load_saved_config(path: str | Path) -> dict:
    path = absolute_path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def make_time_bins(eps_time: float, requested: tuple[float, ...]) -> list[tuple[float, float]]:
    edges = sorted({float(edge) for edge in requested if float(edge) > eps_time})
    edges = [eps_time, *edges]
    if edges[-1] < 1.0:
        edges.append(1.0)
    return [(lo, hi) for lo, hi in zip(edges, edges[1:]) if hi > lo]


def normalized_dataset(saved: dict, split: str) -> torch.Tensor:
    level = int(saved["wavelet_level"])
    data_path = (
        absolute_path(saved["store_path_dataset"])
        / "processed"
        / f"j{level}_{split}.pt"
    )
    data = load_wave_tensor(data_path, expected_channels=int(saved["prior_channels"]) + int(saved["target_channels"]))
    mean = torch.as_tensor(saved["coeff_mean"])
    std = torch.as_tensor(saved["coeff_std"])
    return normalize_with_stats(data, mean, std)


@torch.no_grad()
def evaluate_bin(model, loader, device, lo: float, hi: float, max_batches: int | None) -> dict:
    pred_loss = 0.0
    x0_loss = 0.0
    n_total = 0
    for batch_index, (coeffs,) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        coeffs = coeffs.to(device)
        prior = coeffs[:, :model.prior_channels]
        details = coeffs[:, model.prior_channels:]
        n = details.shape[0]
        t = lo + (hi - lo) * torch.rand(n, device=device, dtype=details.dtype)
        eps = torch.randn_like(details)
        noisy = model(details, t, eps)
        prediction = model.network(torch.cat((prior, noisy), dim=1), t)
        target = model.prediction_target(details, t, eps)
        x0_hat = model.predict_details0_from_prediction(noisy, t, prior, prediction)
        pred_loss += (prediction - target).square().mean(dim=(1, 2, 3)).sum().item()
        x0_loss += (x0_hat - details).square().mean(dim=(1, 2, 3)).sum().item()
        n_total += n
    return {
        "n": n_total,
        "prediction_mse": pred_loss / max(n_total, 1),
        "x0_mse": x0_loss / max(n_total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bins", type=float, nargs="*", default=DEFAULT_BINS)
    args = parser.parse_args()

    saved = load_saved_config(args.config_path)
    if args.checkpoint:
        saved["store_path"] = args.checkpoint

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device(args.device)

    data = normalized_dataset(saved, args.split)
    loader = DataLoader(TensorDataset(data), batch_size=args.batch_size, shuffle=False)
    model = build_sgm(
        saved,
        tuple(saved["coeff_chw"][1:]),
        saved["coeff_mean"],
        saved["coeff_std"],
        device,
    )
    model.load_state_dict(torch.load(absolute_path(saved["store_path"]), map_location=device, weights_only=True))
    model.eval()

    print(
        f"level=j{saved['wavelet_level']} split={args.split} "
        f"eps_time={saved['eps_time']} checkpoint={saved['store_path']}"
    )
    for lo, hi in make_time_bins(float(saved["eps_time"]), tuple(args.bins)):
        metrics = evaluate_bin(model, loader, device, lo, hi, args.max_batches)
        print(
            f"t in [{lo:.1e}, {hi:.1e}]: n={metrics['n']} "
            f"pred_mse={metrics['prediction_mse']:.6e} "
            f"x0_mse={metrics['x0_mse']:.6e}"
        )


if __name__ == "__main__":
    main()
