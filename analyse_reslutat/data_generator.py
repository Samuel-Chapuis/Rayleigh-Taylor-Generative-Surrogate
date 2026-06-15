import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Ajoute la racine du projet, quel que soit le dossier courant du terminal.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.diffusion_lib.DDPM import DDPM
from lib.diffusion_lib.UNet import UNet

DATA_SIZE = 1000

# Parametre du modele generatif
MODEL_PT = SCRIPT_DIR / "res" / "RT28-n1000.pt"
DEVICE = "cpu"
N_STEPS = 1000
TIME_EMB_DIM = 100
MIN_BETA = 1e-4
MAX_BETA = 0.02


state = torch.load(MODEL_PT, map_location=DEVICE)
if isinstance(state, dict) and not hasattr(state, "image_chw"):
    ddpm = DDPM(
        UNet(n_steps=N_STEPS, time_emb_dim=TIME_EMB_DIM),
        n_steps=N_STEPS,
        min_beta=MIN_BETA,
        max_beta=MAX_BETA,
        device=DEVICE,
    )
    ddpm.load_state_dict(state)
else:
    ddpm = state

if hasattr(ddpm, "to"):
    ddpm = ddpm.to(DEVICE)
ddpm.eval(); # Le ; est pour éviter un warning de PyTorch sur le mode evaluation


def load_pt_results(ddpm, device, n_samples=8):
    """
    Permet de générer des images à partir d'un modèle DDPM et de les retourner sous forme de tenseur.

    Args:
        ddpm: Modèle DDPM utilisé pour la génération.
        device: Périphérique sur lequel générer les images.
        n_samples (int, optional): Nombre d'images a generer. Par défaut à 8.
    """

    # On génère pas à pas et on snapshote a t finale.
    c, h, w = ddpm.image_chw
    x = torch.randn(n_samples, c, h, w).to(device)

    with torch.no_grad():
        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)
            alpha_t, alpha_t_bar = ddpm.alphas[t], ddpm.alpha_bars[t]
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)
            if t > 0:
                x = x + ddpm.betas[t].sqrt() * torch.randn_like(x)
            step_pct = t / ddpm.n_steps
            images = x.cpu()

    return images

def generate_ddpm_dataset(model, n_samples=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Generation des images avec le DDPM...")
    return load_pt_results(model, device=device, n_samples=n_samples)


generated_dataset = generate_ddpm_dataset(ddpm, n_samples=DATA_SIZE)

ddpmDumb = DDPM(
    UNet(n_steps=N_STEPS, time_emb_dim=TIME_EMB_DIM),
    n_steps=N_STEPS,
    min_beta=MIN_BETA,
    max_beta=MAX_BETA,
    device=DEVICE,
)

noise_dataset = generate_ddpm_dataset(ddpmDumb, n_samples=DATA_SIZE)

GENERATED_DIR = SCRIPT_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
torch.save(generated_dataset, GENERATED_DIR / "generated_dataset.pt")
torch.save(noise_dataset, GENERATED_DIR / "noise_dataset.pt")
