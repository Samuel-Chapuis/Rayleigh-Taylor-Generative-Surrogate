import torch
import torch.nn as nn

class DDPM(nn.Module): 
    """
    DDPM est une classe qui implémente un modèle de diffusion probabiliste pour la génération d'images. 
    Il ne s'agit pas d'un réseau de neurones complet, mais plutôt d'une structure qui gère les étapes de diffusion et de débruitage.

    Args:
        nn (Module): classe de base pour les modules de réseau de neurones dans PyTorch.
    """    

    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 28, 28)):
        """
        Classe d'implémentation d'un modèle de diffusion probabiliste (DDPM).

        Args:
            network (_type_): _description_
            n_steps (int, optional): _description_. Defaults to 200.
            min_beta (_type_, optional): _description_. Defaults to 10**-4.
            max_beta (float, optional): _description_. Defaults to 0.02.
            device (_type_, optional): _description_. Defaults to None.
            image_chw (tuple, optional): _description_. Defaults to (1, 28, 28).
        """        
        super(DDPM, self).__init__()
        self.n_steps = n_steps              # Nombre total d'étapes de diffusion (plus il est grand, plus la diffusion est fine)
        self.device = device
        self.image_chw = image_chw          # Format des images (channels, height, width)
        self.network = network.to(device)   # Réseau de neurones qui va apprendre à prédire le bruit

        # Création d'une séquence de betas (variance du bruit ajouté à chaque étape)
        # On interpole linéairement entre min_beta et max_beta
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        
        self.alphas = 1 - self.betas        # alpha = 1 - beta (quantité d'information conservée à chaque étape)

        # alpha_bar = produit cumulatif des alphas
        # Cela représente combien de signal original reste après t étapes
        self.alpha_bars = torch.tensor(
            [torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]
        ).to(device)

    def forward(self, x0, t, eta=None):
        """
        Forward process (diffusion):
        On ajoute du bruit à une image propre x0 jusqu'à un temps t donné.

        x0 : image originale
        t : timestep (peut être un batch de valeurs)
        eta : bruit (si None, généré aléatoirement)
        """

        n, c, h, w = x0.shape           # Dimensions du batch
        a_bar = self.alpha_bars[t]      # Récupère alpha_bar pour chaque t

        # Si aucun bruit n'est fourni, on génère du bruit gaussien
        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        # Formule clé du DDPM :
        # x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * bruit
        noisy = (
            a_bar.sqrt().reshape(n, 1, 1, 1) * x0
            + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        )

        return noisy

    def backward(self, x, t):
        """
        Reverse process (débruitage):
        Le réseau essaie de prédire le bruit présent dans l'image x au temps t.

        x : image bruitée
        t : timestep

        Retour :
        estimation du bruit
        """
        return self.network(x, t)

    def sample(self, n_samples=16, device=None, c=None, h=None, w=None):
        """
        Génère des images en partant d'un bruit gaussien et en appliquant le processus inverse.

        Args:
            n_samples (int, optional): Nombre d'images à générer. Defaults to 16.
            device (_type_, optional): Périphérique de calcul. Defaults to None.
            c (int, optional): Nombre de canaux des images générées. Defaults to self.image_chw[0].
            h (int, optional): Hauteur des images générées. Defaults to self.image_chw[1].
            w (int, optional): Largeur des images générées. Defaults to self.image_chw[2].

        Returns:
            torch.Tensor: Lot d'images générées.
        """
        with torch.no_grad():
            if device is None:
                device = self.device

            if c is None or h is None or w is None:
                c, h, w = self.image_chw

            x = torch.randn(n_samples, c, h, w).to(device)

            for t in reversed(range(self.n_steps)):
                time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
                eta_theta = self.backward(x, time_tensor)

                alpha_t = self.alphas[t]
                alpha_t_bar = self.alpha_bars[t]

                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                if t > 0:
                    beta_t = self.betas[t]
                    x = x + beta_t.sqrt() * torch.randn(n_samples, c, h, w).to(device)

        return x