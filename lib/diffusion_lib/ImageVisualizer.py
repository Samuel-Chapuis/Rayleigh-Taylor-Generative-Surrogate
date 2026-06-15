import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class ImageVisualizer:
    """
    Utilitaire de visualisation pour afficher des lots d'images et suivre la
    diffusion directe et inverse.
    """    

    def __init__(self, figsize=(8, 8), cmap="gray", output_dir=None):
        """
        Initialise le visualiseur.

        Args:
            figsize (tuple, optional): Taille de la figure Matplotlib utilisée pour l'affichage.
            cmap (str, optional): Palette utilisée pour les images en niveaux de gris.
            output_dir (str | None, optional): Dossier où sauvegarder les figures générées.
        """        
        self.figsize = figsize
        self.cmap = cmap
        self.output_dir = output_dir

    def _save_figure(self, fig, filename):
        """
        Sauvegarde une figure si un répertoire de sortie a été configuré.

        Args:
            fig (matplotlib.figure.Figure): Figure à sauvegarder.
            filename (str): Nom du fichier de sortie.
        """
        if not self.output_dir:
            return

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / filename, bbox_inches="tight", dpi=300)

    def _to_numpy(self, images):
        """
        Convertit un tenseur PyTorch en tableau NumPy si nécessaire.

        Args:
            images: Tenseur PyTorch ou tableau déjà compatible NumPy.

        Returns:
            Les données sous forme de tableau NumPy, ou l'entrée d'origine si elle l'est déjà.
        """
        if isinstance(images, torch.Tensor):
            return images.detach().cpu().numpy()
        return images


    def show_images(self, images, title=""):
        """
        Affiche un lot d'images dans une grille.

        Args:
            images: Lot d'images au format tenseur ou NumPy, attendu sous la
                forme ``(N, C, H, W)``.
            title (str, optional): Titre affiché en haut de la figure.
        """
        images = self._to_numpy(images)

        fig = plt.figure(figsize=self.figsize)

        rows = int(len(images) ** 0.5)
        cols = round(len(images) / rows)

        idx = 0
        for r in range(rows):
            for c in range(cols):
                fig.add_subplot(rows, cols, idx + 1)

                if idx < len(images):
                    plt.imshow(images[idx][0], cmap=self.cmap)
                    plt.axis("off")
                    idx += 1

        fig.suptitle(title, fontsize=20)
        plt.close(fig)
        self._save_figure(fig, f"{title.replace(' ', '_')}.png")
        

    def show_first_batch(self, loader):
        """
        Affiche le premier lot produit par un DataLoader.

        Args:
            loader: Itérateur PyTorch fournissant des lots sous la forme
                ``(images, labels, ...)``.
        """
        for batch in loader:
            self.show_images(batch[0], "Images in the first batch")
            break


    def show_forward(self, ddpm, loader, device):
        """
        Affiche des images bruitées à plusieurs niveaux du processus direct.

        Args:
            ddpm: Modèle DDPM fournissant le processus de diffusion directe.
            loader: DataLoader contenant des images propres.
            device: Périphérique sur lequel exécuter les calculs.
        """


        percentages = [0, 0.25, 0.5, 0.75, 1.0]
        labels = ["t = 0\n(original)", "t = 25%", "t = 50%", "t = 75%", "t = 100%"]
        
        for batch in loader:
            imgs = batch[0]
            n_cols = min(len(imgs), 8)  # max 8 images par ligne

            # Collecter les blocs d'images
            blocks = []
            # Images originales
            blocks.append(imgs[:n_cols])
            # Images bruitées
            with torch.no_grad():
                imgs_device = imgs.to(device)
                for percent in percentages[1:]:
                    noisy = ddpm(
                        imgs_device,
                        [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]
                    )
                    blocks.append(noisy[:n_cols].detach().cpu())

            # --- Figure ---
            n_rows = len(percentages)
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * 1.4, n_rows * 1.4),
                gridspec_kw={"hspace": 0.08, "wspace": 0.04}
            )

            for row_idx, (block, label) in enumerate(zip(blocks, labels)):
                for col_idx in range(n_cols):
                    ax = axes[row_idx, col_idx]
                    img = block[col_idx].detach().cpu()

                    # Normalisation [0,1] pour l'affichage
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    img_np = img.permute(1, 2, 0).numpy() if img.ndim == 3 else img.numpy()

                    # Niveaux de gris ou couleur
                    if img_np.shape[-1] == 1:
                        ax.imshow(img_np.squeeze(), cmap="gray", vmin=0, vmax=1)
                    else:
                        ax.imshow(np.clip(img_np, 0, 1))

                    ax.axis("off")

                    # Étiquette de ligne sur la première colonne uniquement
                    if col_idx == 0:
                        ax.annotate(
                            label,
                            xy=(0, 0.5),
                            xycoords="axes fraction",
                            xytext=(-8, 0),
                            textcoords="offset points",
                            fontsize=7.5,
                            fontfamily="serif",
                            ha="right",
                            va="center",
                            rotation=45,
                            annotation_clip=False
                        )
            # Titre général
            fig.suptitle(
                "DDPM — Forward diffusion process",
                fontsize=10,
                fontfamily="serif",
                y=1.01
            )

            self._save_figure(fig, "ddpm_forward.png")
            plt.close(fig)
            break


    def show_backward(self, ddpm, device, n_samples=8):
        """
        Affiche l'évolution de la génération pendant le processus inverse.

        Args:
            ddpm: Modèle DDPM utilisé pour la génération.
            device: Périphérique sur lequel générer les images.
            n_samples (int, optional): Nombre d'images affichées par ligne.
        """
        percentages = [1.0, 0.75, 0.5, 0.25, 0]
        labels = ["t = 100%\n(bruit)", "t = 75%", "t = 50%", "t = 25%", "t = 0\n(généré)"]

        # On génère pas à pas et on snapshote aux % voulus
        c, h, w = ddpm.image_chw
        x = torch.randn(n_samples, c, h, w).to(device)
        snap_steps = {int(p * ddpm.n_steps): p for p in percentages}
        blocks = {1.0: x.cpu()}  # snapshot initial

        with torch.no_grad():
            for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
                time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
                eta_theta = ddpm.backward(x, time_tensor)
                alpha_t, alpha_t_bar = ddpm.alphas[t], ddpm.alpha_bars[t]
                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)
                if t > 0:
                    x = x + ddpm.betas[t].sqrt() * torch.randn_like(x)
                step_pct = t / ddpm.n_steps
                for p in [0.75, 0.5, 0.25]:
                    if step_pct <= p and p not in blocks:
                        blocks[p] = x.cpu()
            blocks[0] = x.cpu()

        fig, axes = plt.subplots(len(percentages), n_samples, figsize=(n_samples * 1.4, len(percentages) * 1.4),
                                gridspec_kw={"hspace": 0.08, "wspace": 0.04})
        for row_idx, p in enumerate(percentages):
            block = blocks[p]
            for col_idx in range(n_samples):
                ax = axes[row_idx, col_idx]
                img = block[col_idx]
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax.imshow(img.squeeze(), cmap="gray")
                ax.axis("off")
                if col_idx == 0:
                    ax.annotate(labels[row_idx], xy=(0, 0.5), xycoords="axes fraction",
                                xytext=(-8, 0), textcoords="offset points",
                                fontsize=7.5, ha="right", va="center", rotation=45, annotation_clip=False)
        fig.suptitle("DDPM — Backward diffusion process", fontsize=10, y=1.01)
        self._save_figure(fig, "ddpm_backward.png")
        plt.close(fig)
