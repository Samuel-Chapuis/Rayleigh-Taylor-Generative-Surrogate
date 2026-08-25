import torch
import torch.nn as nn

class WaveletConditionalDDPM(nn.Module):
    """
    DDPM conditionnel pour coefficients d'ondelettes.

    Le canal 0 est le prior basse frequence cA. Les canaux 1..3 sont les details
    cH/cV/cD sur lesquels on applique la diffusion et dont le bruit est predit.
    """

    def __init__(
        self,
        network,
        n_steps=1000,
        min_beta=1e-4,
        max_beta=0.02,
        device=None,
        prior_channels=1,
        target_channels=3,
        image_hw=(32, 32),
        coeff_mean=None,
        coeff_std=None,
    ):
        """
        Initialise un DDPM conditionnel sur un prior wavelet basse frequence.

        Args:
            network: Reseau qui predit le bruit sur les canaux de details.
            n_steps: Nombre de pas du processus de diffusion.
            min_beta: Valeur initiale du calendrier lineaire des variances.
            max_beta: Valeur finale du calendrier lineaire des variances.
            device: Device PyTorch utilise pour le modele et les tenseurs.
            prior_channels: Nombre de canaux conserves comme condition.
            target_channels: Nombre de canaux de details diffuses.
            image_hw: Resolution spatiale des coefficients wavelet.
            coeff_mean: Moyenne canal par canal pour denormaliser les coefficients.
            coeff_std: Ecart type canal par canal pour normaliser les coefficients.
        """
        super().__init__()
        self.n_steps = n_steps
        self.device = device
        self.prior_channels = prior_channels
        self.target_channels = target_channels
        self.image_hw = image_hw
        self.network = network.to(device)

        self.register_buffer("betas", torch.linspace(min_beta, max_beta, n_steps))
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

        if coeff_mean is None:
            coeff_mean = torch.zeros(prior_channels + target_channels)
        if coeff_std is None:
            coeff_std = torch.ones(prior_channels + target_channels)
        self.register_buffer("coeff_mean", coeff_mean.float().reshape(1, -1, 1, 1))
        self.register_buffer("coeff_std", coeff_std.float().reshape(1, -1, 1, 1))

    def forward(self, details_0, t, eta=None):
        """
        Applique le processus direct uniquement aux canaux de details.

        Args:
            details_0: Details propres de forme ``(N, target_channels, H, W)``.
            t: Pas de diffusion pour chaque element du batch.
            eta: Bruit gaussien optionnel. Si ``None``, il est echantillonne.

        Returns:
            Details bruites au pas ``t``.
        """
        n, c, h, w = details_0.shape
        if c != self.target_channels:
            raise ValueError(f"Details attendus avec {self.target_channels} canaux, recu {c}.")

        if eta is None:
            eta = torch.randn(n, c, h, w, device=self.device)

        t = self._flatten_time(t)
        a_bar = self.alpha_bars[t].reshape(n, 1, 1, 1)
        return a_bar.sqrt() * details_0 + (1 - a_bar).sqrt() * eta

    def backward(self, noisy_details, t, prior):
        """
        Predit le bruit des details conditionnellement au prior.

        Args:
            noisy_details: Details bruites de forme ``(N, target_channels, H, W)``.
            t: Pas de diffusion associes au batch.
            prior: Canaux conditionnants, typiquement ``cA``.

        Returns:
            Estimation du bruit ajoute aux details.
        """
        model_input = torch.cat((prior, noisy_details), dim=1)
        return self.network(model_input, t)

    def sample(self, prior, device=None):
        """
        Genere des details par diffusion inverse conditionnee par le prior.

        Args:
            prior: Tenseur conditionnant de forme ``(N, prior_channels, H, W)``.
            device: Device optionnel pour l'echantillonnage.

        Returns:
            Coefficients concatenes ``(prior, details_generes)``.
        """
        if device is None:
            device = self.device

        prior = prior.to(device)
        n, c, h, w = prior.shape
        if c != self.prior_channels:
            raise ValueError(f"Prior attendu avec {self.prior_channels} canaux, recu {c}.")

        details = torch.randn(n, self.target_channels, h, w, device=device)
        with torch.no_grad():
            for t in reversed(range(self.n_steps)):
                time_tensor = torch.full((n, 1), t, device=device, dtype=torch.long)
                eta_theta = self.backward(details, time_tensor, prior)

                alpha_t = self.alphas[t]
                alpha_t_bar = self.alpha_bars[t]
                details = (1 / alpha_t.sqrt()) * (
                    details - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
                )

                if t > 0:
                    details = details + self.betas[t].sqrt() * torch.randn_like(details)

        return torch.cat((prior, details), dim=1)

    def denormalize_coeffs(self, coeffs):
        """Repasse des coefficients normalises vers l'echelle physique/statistique."""
        return coeffs * self.coeff_std + self.coeff_mean

    def normalize_coeffs(self, coeffs):
        """Normalise des coefficients avec les statistiques stockees dans le modele."""
        return (coeffs - self.coeff_mean) / self.coeff_std

    def _flatten_time(self, t):
        """Convertit un pas de temps scalaire ou tensoriel en vecteur ``long``."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        return t.to(self.device).long().reshape(-1)
