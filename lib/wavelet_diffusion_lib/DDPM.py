import torch
import torch.nn as nn

class DDPM(nn.Module): 
    """
    Modèle de diffusion probabiliste pour la génération d'images.

    Cette classe encapsule le processus direct de diffusion, le processus inverse
    appris par le réseau de neurones, ainsi que l'échantillonnage d'images à partir
    d'un bruit gaussien.
    """    

    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 28, 28)):
        """
        Initialise un modèle DDPM.

        Args:
            network (torch.nn.Module): Réseau chargé de prédire le bruit à partir
                d'une image bruitée et d'un pas de temps.
            n_steps (int, optional): Nombre total d'étapes de diffusion. Par défaut 200.
            min_beta (float, optional): Valeur minimale du calendrier de bruit.
                Par défaut 10**-4.
            max_beta (float, optional): Valeur maximale du calendrier de bruit.
                Par défaut 0.02.
            device (torch.device | None, optional): Dispositif de calcul à utiliser.
                Par défaut None.
            image_chw (tuple[int, int, int], optional): Format des images sous la forme
                (canaux, hauteur, largeur). Par défaut (1, 28, 28).
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
        Applique le processus direct de diffusion.

        Cette méthode ajoute du bruit à une image propre `x0` jusqu'au pas `t`.

        Args:
            x0 (torch.Tensor): Images propres de forme ``(N, C, H, W)``.
            t (torch.Tensor): Pas de temps pour chaque élément du lot.
            eta (torch.Tensor | None, optional): Bruit à injecter. Si `None`, un
                bruit gaussien est généré automatiquement.

        Returns:
            torch.Tensor: Images bruitées de même forme que `x0`.
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
        Applique le processus inverse appris par le réseau.

        Args:
            x (torch.Tensor): Images bruitées.
            t (torch.Tensor): Pas de temps associé à chaque image.

        Returns:
            torch.Tensor: Estimation du bruit prédit par le réseau.
        """
        return self.network(x, t)

    def sample(self, n_samples=16, device=None, c=None, h=None, w=None):
        """
        Génère des images en partant d'un bruit gaussien.

        Args:
            n_samples (int, optional): Nombre d'images à générer. Par défaut 16.
            device (torch.device | None, optional): Dispositif de calcul à utiliser.
                Par défaut None.
            c (int | None, optional): Nombre de canaux des images générées.
                Par défaut la première valeur de `image_chw`.
            h (int | None, optional): Hauteur des images générées. Par défaut la
                deuxième valeur de `image_chw`.
            w (int | None, optional): Largeur des images générées. Par défaut la
                troisième valeur de `image_chw`.

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