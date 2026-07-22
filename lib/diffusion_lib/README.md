# diffusion_lib

Librairie generique pour les modeles de diffusion DDPM sur images/champs tenseurs.

Modules principaux :

- `DDPM.py` : processus DDPM generique.
- `UNet.py` : reseau U-Net utilise pour predire le bruit.
- `training_loop.py` : boucle d'entrainement DDPM generique.
- `data_loader.py` : loader standard pour datasets pretraites image-like.
- `schedules.py` : utilitaires de schedule de diffusion.
- `Logger.py`, `ImageVisualizer.py`, `utils.py` : outils communs.
