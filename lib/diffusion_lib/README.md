# diffusion_lib

Librairie generique pour les modeles de diffusion DDPM et score-based sur images/champs tenseurs.

Modules principaux :

- `DDPM.py` : processus DDPM generique.
- `SGM.py` : VP-SDE continu, denoising score matching et reverse SDE.
- `UNet.py` : U-Net commun; `continuous_time=True` active l'embedding continu du SGM.
- `training_loop.py` : boucles d'entrainement DDPM et SGM par minibatches.
- `SGM_Foward.py`, `SGM_Generator.py` : workflows d'entrainement et de generation SGM.
- `data_loader.py` : loader standard pour datasets pretraites image-like.
- `schedules.py` : utilitaires de schedule de diffusion.
- `Logger.py`, `ImageVisualizer.py`, `utils.py` : outils communs.
