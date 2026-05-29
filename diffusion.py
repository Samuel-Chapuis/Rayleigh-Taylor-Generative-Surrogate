# %%
from dataclasses import dataclass
import random
import numpy as np
from tqdm.auto import tqdm
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchview import draw_graph

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST


from diffusion_lib.ImageVisualizer import ImageVisualizer
from diffusion_lib.Logger import Logger
from diffusion_lib.data_loader import data_loader
from diffusion_lib.utils import *
from diffusion_lib.DDPM import *
from diffusion_lib.UNet import *
from diffusion_lib.embeding import *
from diffusion_lib.training_loop import *
# %%

@dataclass(frozen=True)
class Config:
    device: torch.device = None

    # Parametres généraux
    seed: int = 0
    store_path_dataset: str = "data/MNIST"
    viz: ImageVisualizer = ImageVisualizer(output_dir="outputs/img")
    no_train: bool = False
    batch_size: int = 128
    n_epochs: int = 1
    lr: float = 0.001
    store_path: str = "outputs/model/ddpm_mnist.pt"
    input_path: str = ""
    log_path: str = "outputs/logs/ddpm_mnist.log"
    csv_path: str = "outputs/logs/ddpm_mnist.csv"

    # Hyperparametres
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    out_channels: int = 10

    # Parametres du DDPM
    time_emb_dim: int = 100 # dimension de l'embedding temporel
    n_steps: int = 1000 # discretisation du processus de diffusion (nombre de bruitages successifs)
    
    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        object.__setattr__(self, "device", get_best_device())
        print(self.device)

# %%
''' Setup '''
# Preparation de l'expérience
config = Config()
logger = Logger(config.log_path, config.csv_path)
logger.log_experiment_start({
    "seed": config.seed,
    "store_path_dataset": config.store_path_dataset,
    "batch_size": config.batch_size,
    "n_epochs": config.n_epochs,
    "lr": config.lr,
    "store_path": config.store_path,
    "time_emb_dim": config.time_emb_dim,
    "n_steps": config.n_steps,
    "device": config.device,
})


loader = data_loader(config)


# On affiche le premier batch de donnée pour vérifier que tout est en ordre
config.viz.show_first_batch(loader)


# %%
''' Visualisation '''
# Visualisation du bloc qui va garder la dimension spatiale inchangé. Ce bloque sera utilisé dans tous les niveau du U-Net
block = Block(shape=(1, 28, 28), in_c=1, out_c=10)
model_graph = draw_graph(
    block,
    input_size=(1, 1, 28, 28),  # (batch, C, H, W)
    device='meta'               # pas de mémoire allouée
)
model_graph.visual_graph

# Visualisation du U-Net complet
unet = UNet(n_steps=config.n_steps, time_emb_dim=config.time_emb_dim)
model_graph = draw_graph(
    unet,
    input_data=[
        torch.randn(1, 1, 28, 28),          # x
        torch.zeros(1, 1, dtype=torch.long)  # t
    ],
    device='meta',
    expand_nested=True,   # déroule les sous-modules (MyBlock, etc.)
    show_shapes=True,     # affiche les dimensions des tenseurs
    depth=3               # profondeur du graphe (ajuste selon le détail voulu)
)
model_graph.resize_graph(0.2)
model_graph.visual_graph
# %%

''' Entraînement '''
# Visualisation du processus de diffusion directe avant entraînement
n_steps, min_beta, max_beta = config.n_steps, 10 ** -4, 0.02  # Originally used by the authors
ddpm = DDPM(UNet(n_steps=config.n_steps, time_emb_dim=config.time_emb_dim), n_steps=config.n_steps, min_beta=min_beta, max_beta=max_beta, device=config.device)
if config.input_path:
    if not os.path.exists(config.input_path):
        raise FileNotFoundError(f"Checkpoint introuvable: {config.input_path}")
    ddpm.load_state_dict(torch.load(config.input_path, map_location=config.device))
config.viz.show_forward(ddpm, loader, config.device)

# Visualisation du processus de diffusion inverse avant entraînement
generate = ddpm.sample()
config.viz.show_images(generate, "before training")

# Choix de l'optimiseur
optimizer = Adam(ddpm.parameters(), lr=config.lr)

# Entraînement du modèle
# loss = training_loop(ddpm, loader, config.n_epochs, optimizer, config.device, store_path=config.store_path, logger=logger)

# Chargement du meilleur modèle sauvegardé
best_model = DDPM(UNet(n_steps=config.n_steps, time_emb_dim=config.time_emb_dim), n_steps=config.n_steps, min_beta=min_beta, max_beta=max_beta, device=config.device)
best_model.load_state_dict(torch.load(config.store_path, map_location=config.device))
best_model.eval()

# Visualisation du processus de diffusion inverse après entraînement
config.viz.show_backward(best_model, config.device)