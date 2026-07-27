import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytorch_fid_src = PROJECT_ROOT / "lib" / "pytorch-fid-master" / "src"
if str(pytorch_fid_src) not in sys.path:
    sys.path.insert(0, str(pytorch_fid_src))

from lib.diffusion_lib.DDPM import DDPM
from lib.diffusion_lib.UNet import UNet


DATA_SIZE = 16
CONFIG_PATH = PROJECT_ROOT / "outputs" / "model" / "RT64_config.json"
GENERATED_DIR = PROJECT_ROOT / "outputs" / "generated"
GENERATED_DATASET_PATH = GENERATED_DIR / "64generated_dataset.pt"
NOISE_DATASET_PATH = GENERATED_DIR / "64noise_dataset.pt"


def load_generation_config(config_path=CONFIG_PATH):
    """
    Charge la configuration produite par diffusion.py.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config introuvable: {config_path}. Lance diffusion.py pour générer ce fichier."
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    config["image_chw"] = tuple(config.get("image_chw", (1, 28, 28)))
    return config


def build_ddpm_from_config(config, device):
    """
    Reconstruit un DDPM avec les mêmes hyperparamètres que pendant l'entraînement.
    """
    image_chw = tuple(config["image_chw"])
    network = UNet(
        n_steps=config["n_steps"],
        time_emb_dim=config["time_emb_dim"],
        size=image_chw[1],
        in_channels=image_chw[0],
        out_channels=config.get("unet_out_channels"),
        depth=config.get("unet_depth", 3),
        blocks_per_level=config.get("unet_blocks_per_level", 3),
        base_channels=config.get("unet_base_channels", 10),
    )
    return DDPM(
        network,
        n_steps=config["n_steps"],
        min_beta=config["min_beta"],
        max_beta=config["max_beta"],
        device=device,
        image_chw=image_chw,
    )


def load_trained_ddpm(config, device):
    """
    Charge le modèle entraîné depuis le chemin indiqué dans config.json.
    """
    model_path = PROJECT_ROOT / config["store_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {model_path}")

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and not hasattr(state, "image_chw"):
        ddpm = build_ddpm_from_config(config, device)
        ddpm.load_state_dict(state)
    else:
        ddpm = state

    if hasattr(ddpm, "to"):
        ddpm = ddpm.to(device)
    ddpm.eval()
    return ddpm


def generate_ddpm_dataset(ddpm, device=None, n_samples=8, batch_size=128):
    """
    Génère un dataset d'images avec un modèle DDPM.

    Args:
        ddpm: Modèle DDPM utilisé pour la génération.
        device: Périphérique sur lequel générer les images.
        n_samples (int): Nombre total d'images à générer.
        batch_size (int): Nombre d'images générées par lot.

    Returns:
        torch.Tensor: Tenseur CPU de forme (n_samples, C, H, W).
    """
    if device is None:
        device = getattr(ddpm, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")

    samples = []
    remaining = n_samples
    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        print(f"Generation de {current_batch_size} images ({n_samples - remaining}/{n_samples} deja generees)")
        with torch.no_grad():
            batch = ddpm.sample(n_samples=current_batch_size, device=device)
        samples.append(batch.cpu())
        remaining -= current_batch_size

    return torch.cat(samples, dim=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_generation_config()

    ddpm = load_trained_ddpm(config, device)
    generated_dataset = generate_ddpm_dataset(ddpm, device=device, n_samples=DATA_SIZE)

    noise_model = build_ddpm_from_config(config, device)
    noise_model.eval()
    noise_dataset = generate_ddpm_dataset(noise_model, device=device, n_samples=DATA_SIZE)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(generated_dataset, GENERATED_DATASET_PATH)
    torch.save(noise_dataset, NOISE_DATASET_PATH)


if __name__ == "__main__":
    main()
