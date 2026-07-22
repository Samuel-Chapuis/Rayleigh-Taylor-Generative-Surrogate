import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.wavelet_diffusion_lib.UNet import UNet
from lib.wavelet_diffusion_lib.ConditionalDDPM import WaveletConditionalDDPM
from _to_move_wave_2D.wave_diffusion import load_wave_tensor, normalize_with_stats

# ============================================================
# Configuration
# ============================================================

DECOMPOSITION = "j1"

CONFIG_PATH = PROJECT_ROOT / f"outputs/model/wave_{DECOMPOSITION}_RT64_config.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs/generated"

OUTPUT_DATASET = OUTPUT_DIR / f"{DECOMPOSITION}_gen_wave_data.pt"

DATASET = "validation"          # train / validation / test

BATCH_SIZE = 128
N_BATCHES = 1
# ============================================================
# Utilitaires
# ============================================================

def load_config():

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    config["coeff_chw"] = tuple(config["coeff_chw"])

    return config


def build_model(config, device):

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
        coeff_mean=torch.tensor(config["coeff_mean"]),
        coeff_std=torch.tensor(config["coeff_std"]),
    )

    checkpoint = torch.load(
        PROJECT_ROOT / config["store_path"],
        map_location=device,
    )

    ddpm.load_state_dict(checkpoint)

    ddpm.to(device)
    ddpm.eval()

    return ddpm


# ============================================================
# Main
# ============================================================

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config()

    ddpm = build_model(config, device)

    dataset_path = (
        Path(config["store_path_dataset"])
        / "processed"
        / f"j{config['wavelet_level']}_{DATASET}.pt"
    )

    coeffs = load_wave_tensor(dataset_path)

    mean = torch.tensor(config["coeff_mean"])
    std = torch.tensor(config["coeff_std"])

    coeffs = normalize_with_stats(coeffs, mean, std)

    loader = DataLoader(
        TensorDataset(coeffs),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    generated = []

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader):

            if batch_idx >= N_BATCHES:
                break

            coeffs = batch[0]

            prior = coeffs[:, : config["prior_channels"]].to(device)

            print(
                f"Batch {batch_idx + 1}/{N_BATCHES} "
                f"({prior.shape[0]} images)"
            )

            fake = ddpm.sample(prior)

            fake = ddpm.denormalize_coeffs(fake)

            generated.append(fake.cpu())

    generated = torch.cat(generated, dim=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(generated, OUTPUT_DATASET)

    print("Dataset sauvegardé :", OUTPUT_DATASET)
    print("Shape :", generated.shape)


if __name__ == "__main__":
    main()
