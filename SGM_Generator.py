"""Generation de datasets avec un SGM entraine.

L'interface et les fichiers produits sont paralleles a ``DDPM_Generator.py``.
"""

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.diffusion_lib.SGM import SGM
from lib.diffusion_lib.UNet import UNet


DATA_SIZE = 16
CONFIG_PATH = PROJECT_ROOT / "outputs" / "model" / "RT64_sgm_config.json"
GENERATED_DIR = PROJECT_ROOT / "outputs" / "generated"
GENERATED_DATASET_PATH = GENERATED_DIR / "64generated_dataset_sgm.pt"
NOISE_DATASET_PATH = GENERATED_DIR / "64noise_dataset_sgm.pt"


def load_generation_config(config_path=CONFIG_PATH):
    """Charge la configuration produite par ``SGM_Foward.py``."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config introuvable: {config_path}. Lance SGM_Foward.py pour la creer."
        )
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    config["image_chw"] = tuple(config.get("image_chw", (1, 28, 28)))
    return config


def build_sgm_from_config(config, device):
    """Reconstruit exactement l'architecture SGM de l'entrainement."""
    image_chw = tuple(config["image_chw"])
    network = UNet(
        n_steps=1000,  # non utilise en mode temps continu
        time_emb_dim=config["time_emb_dim"],
        size=image_chw[1],
        in_channels=image_chw[0],
        out_channels=config.get("unet_out_channels"),
        depth=config.get("unet_depth", 3),
        blocks_per_level=config.get("unet_blocks_per_level", 3),
        base_channels=config.get("unet_base_channels", 32),
        continuous_time=True,
    )
    return SGM(
        network,
        beta_min=config.get("beta_min", 0.1),
        beta_max=config.get("beta_max", 20.0),
        eps_time=config.get("eps_time", 1e-3),
        prediction_type=config.get("prediction_type", "epsilon"),
        device=device,
        image_chw=image_chw,
    )


def load_trained_sgm(config, device):
    """Reconstruit le SGM et charge son ``state_dict``."""
    model_path = Path(config["store_path"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {model_path}")

    sgm = build_sgm_from_config(config, device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    sgm.load_state_dict(state)
    sgm.eval()
    return sgm


def generate_sgm_dataset(
    sgm,
    device=None,
    n_samples=8,
    batch_size=128,
    n_steps=None,
    solver="heun",
    clip_denoised=True,
):
    """Genere ``n_samples`` images par lots et retourne un tenseur CPU."""
    if device is None:
        device = sgm.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if n_samples < 0 or batch_size < 1:
        raise ValueError("n_samples doit etre >= 0 et batch_size doit etre positif.")
    if n_samples == 0:
        c, h, w = sgm.image_chw
        return torch.empty(0, c, h, w)

    n_steps = 1000 if n_steps is None else n_steps
    samples = []
    remaining = n_samples
    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        print(
            f"Generation de {current_batch_size} images "
            f"({n_samples - remaining}/{n_samples} deja generees)"
        )
        batch = sgm.sample(
            n_samples=current_batch_size,
            device=device,
            n_steps=n_steps,
            solver=solver,
            clip_denoised=clip_denoised,
        )
        samples.append(batch.cpu())
        remaining -= current_batch_size
    return torch.cat(samples, dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_generation_config()
    sampling_steps = config.get("sampling_steps", 1000)
    solver = config.get("sampler", "heun")
    clip_denoised = config.get("clip_denoised", True)

    sgm = load_trained_sgm(config, device)
    generated_dataset = generate_sgm_dataset(
        sgm,
        device=device,
        n_samples=DATA_SIZE,
        n_steps=sampling_steps,
        solver=solver,
        clip_denoised=clip_denoised,
    )

    # Reference utile pour comparer l'effet du reseau appris.
    noise_model = build_sgm_from_config(config, device)
    noise_model.eval()
    noise_dataset = generate_sgm_dataset(
        noise_model,
        device=device,
        n_samples=DATA_SIZE,
        n_steps=sampling_steps,
        solver=solver,
        clip_denoised=clip_denoised,
    )

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(generated_dataset, GENERATED_DATASET_PATH)
    torch.save(noise_dataset, NOISE_DATASET_PATH)
    print(f"Dataset genere sauvegarde dans {GENERATED_DATASET_PATH}")
    print(f"Dataset de reference sauvegarde dans {NOISE_DATASET_PATH}")


if __name__ == "__main__":
    main()
