import json
import sys
from pathlib import Path

import numpy as np
import pywt
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.wavelet_diffusion_lib.UNet import UNet
from lib.wavelet_diffusion_lib.ConditionalDDPM import WaveletConditionalDDPM
from _to_move_wave_2D.wave_diffusion import load_wave_tensor

# ============================================================
# Configuration
# ============================================================

# The cascade must run from the coarsest decomposition to the finest one.
LEVELS = [3, 2, 1]

OUTPUT_DIR = PROJECT_ROOT / "outputs/generated/cascade"
DATASET = "validation"  # training / validation / test: adapt to your filenames

BATCH_SIZE = 64
N_BATCHES = 1

WAVELET = "db1"
BORDER_MODE = "periodization"

# Expected channel convention throughout this script:
#   coefficient tensor = [cA, cH, cV, cD]
#   generated target   = [cH, cV, cD]


def load_config(level: int) -> dict:
    path = PROJECT_ROOT / f"outputs/model/wave_j{level}_RT64_config.json"
    with path.open("r") as file:
        config = json.load(file)

    config["coeff_chw"] = tuple(config["coeff_chw"])

    configured_level = int(config["wavelet_level"])
    if configured_level != level:
        raise ValueError(
            f"Configuration mismatch: requested j{level}, "
            f"but {path.name} declares wavelet_level={configured_level}."
        )

    if int(config["prior_channels"]) != 1:
        raise ValueError("The cascade assumes exactly one prior channel: cA.")
    if int(config["target_channels"]) != 3:
        raise ValueError("The cascade assumes three generated channels: cH, cV, cD.")

    return config


def build_model(config: dict, device: torch.device) -> WaveletConditionalDDPM:
    coeff_chw = config["coeff_chw"]

    network = UNet(
        n_steps=config["n_steps"],
        time_emb_dim=config["time_emb_dim"],
        size=coeff_chw[1],
        in_channels=config["prior_channels"] + config["target_channels"],
        out_channels=config["target_channels"],
        depth=config["unet_depth"],
        blocks_per_level=config["unet_blocks_per_level"],
        base_channels=config["unet_base_channels"],
    )

    ddpm = WaveletConditionalDDPM(
        network,
        n_steps=config["n_steps"],
        min_beta=config["min_beta"],
        max_beta=config["max_beta"],
        device=device,
        prior_channels=config["prior_channels"],
        target_channels=config["target_channels"],
        image_hw=coeff_chw[1:],
        coeff_mean=torch.tensor(config["coeff_mean"], dtype=torch.float32),
        coeff_std=torch.tensor(config["coeff_std"], dtype=torch.float32),
    )

    checkpoint_path = PROJECT_ROOT / config["store_path"]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(checkpoint)
    ddpm.to(device)
    ddpm.eval()

    return ddpm


def normalize_approximation(ca: torch.Tensor, config: dict) -> torch.Tensor:
    """Normalize physical cA values using the cA statistics of this level."""
    if ca.ndim != 4 or ca.shape[1] != 1:
        raise ValueError(f"Expected cA shape [N,1,H,W], got {tuple(ca.shape)}.")

    mean = torch.as_tensor(config["coeff_mean"], dtype=ca.dtype)[0]
    std = torch.as_tensor(config["coeff_std"], dtype=ca.dtype)[0]

    if not torch.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid cA standard deviation at j{config['wavelet_level']}: {std}.")

    return (ca - mean) / std


def inverse_dwt_batch(
    ca: torch.Tensor,
    details: torch.Tensor,
    wavelet: str = WAVELET,
    mode: str = BORDER_MODE,
) -> torch.Tensor:
    """
    Reconstruct the next-finer approximation.

    Parameters
    ----------
    ca:
        Physical approximation coefficients, shape [N,1,H,W].
    details:
        Physical detail coefficients [cH,cV,cD], shape [N,3,H,W].

    Returns
    -------
    torch.Tensor
        Reconstructed physical field cA_(j-1), shape [N,1,2H,2W]
        for db1/periodization and even dimensions.
    """
    if ca.ndim != 4 or ca.shape[1] != 1:
        raise ValueError(f"Expected cA [N,1,H,W], got {tuple(ca.shape)}.")
    if details.ndim != 4 or details.shape[1] != 3:
        raise ValueError(f"Expected details [N,3,H,W], got {tuple(details.shape)}.")
    if ca.shape[0] != details.shape[0] or ca.shape[2:] != details.shape[2:]:
        raise ValueError(
            "cA and detail coefficients must have identical batch and spatial dimensions: "
            f"cA={tuple(ca.shape)}, details={tuple(details.shape)}."
        )

    ca_np = ca.detach().cpu().numpy()
    details_np = details.detach().cpu().numpy()

    reconstructed = []
    for sample_idx in range(ca_np.shape[0]):
        c_a = ca_np[sample_idx, 0]
        c_h = details_np[sample_idx, 0]
        c_v = details_np[sample_idx, 1]
        c_d = details_np[sample_idx, 2]

        field = pywt.idwt2(
            (c_a, (c_h, c_v, c_d)),
            wavelet=wavelet,
            mode=mode,
        )
        reconstructed.append(field)

    result = np.stack(reconstructed, axis=0)[:, None, :, :]
    return torch.from_numpy(result).to(dtype=ca.dtype)


def generate_details_for_level(
    ca_physical: torch.Tensor,
    level: int,
    config: dict,
    ddpm: WaveletConditionalDDPM,
    device: torch.device,
) -> torch.Tensor:
    expected_hw = tuple(config["coeff_chw"][1:])
    if tuple(ca_physical.shape[2:]) != expected_hw:
        raise ValueError(
            f"Spatial mismatch at j{level}: model expects {expected_hw}, "
            f"but current cA has {tuple(ca_physical.shape[2:])}."
        )

    loader = DataLoader(
        TensorDataset(ca_physical),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    generated_batches = []
    with torch.no_grad():
        for batch_index, (ca_batch,) in enumerate(loader):
            prior = normalize_approximation(ca_batch, config).to(device)
            print(
                f"j{level}: batch {batch_index + 1}/{len(loader)} "
                f"({prior.shape[0]} samples, prior={tuple(prior.shape)})"
            )

            sampled_normalized = ddpm.sample(prior)
            sampled_physical = ddpm.denormalize_coeffs(sampled_normalized)

            if sampled_physical.ndim != 4:
                raise RuntimeError(
                    f"j{level} returned a tensor with shape "
                    f"{tuple(sampled_physical.shape)}; expected [N,C,H,W]."
                )

            n_channels = sampled_physical.shape[1]

            if n_channels == 3:
                # Convention 1: the sampler returns only [cH, cV, cD].
                details_physical = sampled_physical

            elif n_channels == 4:
                # Convention 2: the DDPM helper returns the complete tuple
                # [cA, cH, cV, cD].  The approximation used by the cascade
                # remains ca_batch; only the three detail channels are needed
                # for the inverse wavelet transform.
                returned_ca = sampled_physical[:, 0:1]
                details_physical = sampled_physical[:, 1:4]

                if returned_ca.shape != ca_batch.shape:
                    raise RuntimeError(
                        f"j{level} returned cA with shape {tuple(returned_ca.shape)}, "
                        f"while the conditioning cA has shape {tuple(ca_batch.shape)}."
                    )

                # This diagnostic is deliberately non-fatal: depending on the
                # implementation, the returned cA may be the exact condition
                # or a reconstructed/denormalized copy of it.
                max_ca_difference = (returned_ca.cpu() - ca_batch).abs().max().item()
                print(
                    f"j{level}: sampler returned [cA,cH,cV,cD]; "
                    f"using channels 1:4 as details "
                    f"(max |returned cA - input cA|={max_ca_difference:.3e})."
                )

            else:
                raise RuntimeError(
                    f"j{level} returned {n_channels} channels; expected either "
                    "3 ([cH,cV,cD]) or 4 ([cA,cH,cV,cD])."
                )

            generated_batches.append(details_physical.cpu())

    return torch.cat(generated_batches, dim=0)



def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if LEVELS != sorted(LEVELS, reverse=True):
        raise ValueError(f"LEVELS must be in descending order, got {LEVELS}.")
    if any(a - b != 1 for a, b in zip(LEVELS, LEVELS[1:])):
        raise ValueError(f"LEVELS must be consecutive, got {LEVELS}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load only the coarsest dataset. It supplies the initial physical cA_J.
    coarsest_level = LEVELS[0]
    coarsest_config = load_config(coarsest_level)
    dataset_path = (
        Path(coarsest_config["store_path_dataset"])
        / "processed"
        / f"j{coarsest_config['wavelet_level']}_{DATASET}.pt"
    )

    coarsest_coeffs = load_wave_tensor(dataset_path)
    if coarsest_coeffs.ndim != 4 or coarsest_coeffs.shape[1] < 1:
        raise ValueError(
            f"Expected coarsest tensor [N,C,H,W], got {tuple(coarsest_coeffs.shape)}."
        )

    n_samples = min(len(coarsest_coeffs), BATCH_SIZE * N_BATCHES)
    if n_samples == 0:
        raise ValueError("No sample selected. Check BATCH_SIZE, N_BATCHES and the dataset.")

    current_ca = coarsest_coeffs[:n_samples, 0:1].float().cpu()
    torch.save(current_ca, OUTPUT_DIR / f"j{coarsest_level}_initial_cA.pt")
    print(f"Initial cA_j{coarsest_level}: {tuple(current_ca.shape)}")

    for level in LEVELS:
        print(f"\n=== Cascade level j{level} ===")
        config = load_config(level)
        ddpm = build_model(config, device)

        details = generate_details_for_level(
            ca_physical=current_ca,
            level=level,
            config=config,
            ddpm=ddpm,
            device=device,
        )

        # Save both generated details and the complete coefficient tuple.
        full_coefficients = torch.cat((current_ca, details), dim=1)
        torch.save(details, OUTPUT_DIR / f"j{level}_generated_details.pt")
        torch.save(full_coefficients, OUTPUT_DIR / f"j{level}_generated_coefficients.pt")

        print(f"Generated details j{level}: {tuple(details.shape)}")
        print(f"Complete coefficients j{level}: {tuple(full_coefficients.shape)}")

        # This output is cA_(level-1), or the final image after j1.
        current_ca = inverse_dwt_batch(current_ca, details)
        output_name = (
            "generated_images.pt"
            if level == LEVELS[-1]
            else f"j{level - 1}_reconstructed_cA.pt"
        )
        torch.save(current_ca, OUTPUT_DIR / output_name)
        print(f"Inverse DWT output: {tuple(current_ca.shape)} -> {output_name}")

        del ddpm
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nFinal generated images: {tuple(current_ca.shape)}")
    print(f"Saved in: {OUTPUT_DIR / 'generated_images.pt'}")


if __name__ == "__main__":
    main()
