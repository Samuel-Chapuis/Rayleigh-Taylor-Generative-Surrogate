import math

import torch
import torch.nn as nn


class DDPM(nn.Module):
    """Diffusion OU continue de Mallat pour un champ non conditionné.

    Le réseau est entraîné avec ``x_t = exp(-t)x0 + sqrt(1-exp(-2t))*eps``.
    ``sampling_steps`` ne contrôle que la discrétisation numérique du sampler.
    """

    def __init__(
        self, network, final_time=5.0, sampling_steps=16, snr_terminal=None,
        sampling_eta=1.0, device=None, image_chw=(1, 28, 28),
    ):
        super().__init__()
        if final_time <= 0 or sampling_steps < 1:
            raise ValueError("final_time doit être positif et sampling_steps >= 1.")
        if snr_terminal is not None:
            if snr_terminal <= 0:
                raise ValueError("snr_terminal doit être strictement positif.")
            final_time = 0.5 * math.log1p(1.0 / snr_terminal)
        self.final_time = float(final_time)
        self.sampling_steps = int(sampling_steps)
        self.snr_terminal = float(1.0 / math.expm1(2 * self.final_time))
        self.sampling_eta = float(sampling_eta)
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)

    def forward(self, x0, t, eta=None):
        t = torch.as_tensor(t, device=x0.device, dtype=x0.dtype).reshape(-1)
        if t.shape[0] != x0.shape[0]:
            raise ValueError("t doit contenir un temps par élément du batch.")
        if eta is None:
            eta = torch.randn_like(x0)
        alpha = torch.exp(-2.0 * t).reshape(-1, 1, 1, 1)
        return alpha.sqrt() * x0 + (1.0 - alpha).clamp_min(0).sqrt() * eta

    def backward(self, x, t):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(-1)
        return self.network(x, t)

    def sample(self, n_samples=16, device=None, c=None, h=None, w=None):
        device = device or self.device
        c, h, w = c or self.image_chw[0], h or self.image_chw[1], w or self.image_chw[2]
        x = torch.randn(n_samples, c, h, w, device=device)
        return self._sample_tensor(x)

    def _sample_tensor(self, x):
        # Temps décroissants : t_K=T,...,t_0=0. Les 16 évaluations sont
        # indépendantes du nombre de temps utilisés pendant l'entraînement.
        times = torch.linspace(self.final_time, 0.0, self.sampling_steps + 1,
                               device=x.device, dtype=x.dtype)
        with torch.no_grad():
            for i in range(self.sampling_steps):
                t, t_prev = times[i], times[i + 1]
                t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
                eps = self.backward(x, t_batch)
                alpha_t = torch.exp(-2 * t)
                alpha_prev = torch.exp(-2 * t_prev)
                x0_hat = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
                ratio = (alpha_t / alpha_prev).clamp(0, 1)
                sigma = self.sampling_eta * torch.sqrt(
                    ((1 - alpha_prev) / (1 - alpha_t).clamp_min(1e-12))
                    * (1 - ratio)
                )
                residual = (1 - alpha_prev - sigma.square()).clamp_min(0).sqrt()
                x = alpha_prev.sqrt() * x0_hat + residual * eps
                if i < self.sampling_steps - 1:
                    x = x + sigma * torch.randn_like(x)
        return x
