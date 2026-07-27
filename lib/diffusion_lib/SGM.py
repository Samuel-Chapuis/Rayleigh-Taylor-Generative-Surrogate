"""Score-based generative model based on a continuous VP-SDE."""

import torch
import torch.nn as nn


class SGM(nn.Module):
    """VP-SDE continu dont le reseau predit le bruit epsilon.

    Le processus direct est

    ``dx = -0.5 beta(t) x dt + sqrt(beta(t)) dW``

    avec ``beta(t)`` affine. Sa marginale exacte est utilisee pendant
    l'entrainement; aucune simulation Euler-Maruyama n'est donc necessaire.
    Le reverse SDE est integre par Euler-Maruyama dans :meth:`sample`.
    """

    def __init__(
        self,
        network,
        beta_min=0.1,
        beta_max=20.0,
        device=None,
        image_chw=(1, 28, 28),
        eps_time=1e-3,
    ):
        super().__init__()
        if beta_min <= 0 or beta_max < beta_min:
            raise ValueError("Il faut 0 < beta_min <= beta_max.")
        if not 0 < eps_time < 1:
            raise ValueError("eps_time doit appartenir a (0, 1).")

        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.eps_time = float(eps_time)
        self.image_chw = tuple(image_chw)
        self.device = device
        self.network = network.to(device)

    def beta(self, t):
        """Retourne beta(t), avec ``t`` dans [0, 1]."""
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def integrated_beta(self, t):
        """Retourne ``integral_0^t beta(s) ds``."""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t.square()

    def alpha_bar(self, t):
        """Facteur de puissance du signal dans la marginale VP."""
        return torch.exp(-self.integrated_beta(t))

    def marginal_std(self, t):
        return torch.sqrt((1.0 - self.alpha_bar(t)).clamp_min(1e-12))

    def forward(self, x0, t, eps=None):
        """Echantillonne la marginale exacte ``q(x_t | x_0)``."""
        t = t.to(device=x0.device, dtype=x0.dtype).reshape(-1)
        if t.shape[0] != x0.shape[0]:
            raise ValueError("Il faut un temps par element du batch.")
        if eps is None:
            eps = torch.randn_like(x0)
        abar = self.alpha_bar(t).reshape(-1, 1, 1, 1)
        xt = abar.sqrt() * x0 + (1.0 - abar).clamp_min(1e-12).sqrt() * eps
        return xt

    def predict_epsilon(self, x, t):
        """Prediction epsilon du reseau conditionne par un temps continu."""
        return self.network(x, t.reshape(-1))

    def score(self, x, t):
        """Convertit epsilon-prediction en score de la marginale bruitée."""
        t = t.to(device=x.device, dtype=x.dtype).reshape(-1)
        return -self.predict_epsilon(x, t) / self.marginal_std(t).reshape(-1, 1, 1, 1)

    def backward(self, x, t):
        """Alias compatible avec ``DDPM.backward``; retourne le score appris."""
        return self.score(x, t)

    @torch.no_grad()
    def sample(self, n_samples=16, device=None, n_steps=1000, return_history=False):
        """Integre le reverse VP-SDE par Euler-Maruyama.

        ``n_steps`` controle uniquement l'integration de generation, pas
        l'entrainement. Le dernier pas est deterministe pour reduire la variance
        de discretisation a l'approche de t=eps_time.
        """
        device = self.device if device is None else device
        if n_steps < 1:
            raise ValueError("n_steps doit etre >= 1.")
        c, h, w = self.image_chw
        x = torch.randn(n_samples, c, h, w, device=device)
        history = [x.detach().cpu()] if return_history else None
        dt = (1.0 - self.eps_time) / n_steps

        was_training = self.training
        self.eval()
        for i in range(n_steps, 0, -1):
            # Le premier appel est a t=1; l'etat est ensuite avance jusqu'a
            # t=eps_time par pas de taille dt.
            t_value = self.eps_time + i * dt
            t = torch.full((n_samples,), t_value, device=device, dtype=x.dtype)
            beta_t = self.beta(t).reshape(-1, 1, 1, 1)
            score_t = self.score(x, t)
            drift = -0.5 * beta_t * x - beta_t * score_t
            noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            x = x - drift * dt + torch.sqrt(beta_t * dt) * noise
            if return_history:
                history.append(x.detach().cpu())
        if was_training:
            self.train()
        return torch.stack(history) if return_history else x
