"""Score-based generative model based on a continuous VP-SDE."""

import torch
import torch.nn as nn


class SGM(nn.Module):
    """VP-SDE continu avec prediction ``epsilon`` ou ``v``.

    Le processus direct est

    ``dx = -0.5 beta(t) x dt + sqrt(beta(t)) dW``

    avec ``beta(t)`` affine. Sa marginale exacte est utilisee pendant
    l'entrainement; aucune simulation Euler-Maruyama n'est donc necessaire.
    La generation par defaut integre le probability-flow ODE avec Heun.
    """

    def __init__(
        self,
        network,
        beta_min=0.1,
        beta_max=20.0,
        device=None,
        image_chw=(1, 28, 28),
        eps_time=1e-3,
        prediction_type="v",
    ):
        super().__init__()
        if beta_min <= 0 or beta_max < beta_min:
            raise ValueError("Il faut 0 < beta_min <= beta_max.")
        if not 0 < eps_time < 1:
            raise ValueError("eps_time doit appartenir a (0, 1).")

        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.eps_time = float(eps_time)
        if prediction_type not in {"epsilon", "v"}:
            raise ValueError("prediction_type doit etre 'epsilon' ou 'v'.")
        self.prediction_type = prediction_type
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
        """Retourne epsilon reconstruit depuis la parametrisation du reseau."""
        t = t.to(device=x.device, dtype=x.dtype).reshape(-1)
        prediction = self.network(x, t)
        if self.prediction_type == "epsilon":
            return prediction

        # v = alpha * epsilon - sigma * x0 and
        # x_t = alpha * x0 + sigma * epsilon, hence
        # epsilon = sigma * x_t + alpha * v.
        alpha = self.alpha_bar(t).clamp_min(1e-12).sqrt()
        sigma = self.marginal_std(t)
        return sigma.reshape(-1, 1, 1, 1) * x + alpha.reshape(-1, 1, 1, 1) * prediction

    def prediction_target(self, x0, t, eps):
        """Cible du reseau pour la parametrisation choisie."""
        if self.prediction_type == "epsilon":
            return eps
        alpha = self.alpha_bar(t).clamp_min(1e-12).sqrt()
        sigma = self.marginal_std(t)
        return alpha.reshape(-1, 1, 1, 1) * eps - sigma.reshape(-1, 1, 1, 1) * x0

    def score(self, x, t):
        """Convertit epsilon-prediction en score de la marginale bruitée."""
        t = t.to(device=x.device, dtype=x.dtype).reshape(-1)
        return -self.predict_epsilon(x, t) / self.marginal_std(t).reshape(-1, 1, 1, 1)

    def backward(self, x, t):
        """Alias compatible avec ``DDPM.backward``; retourne le score appris."""
        return self.score(x, t)

    @torch.no_grad()
    def sample(
        self,
        n_samples=16,
        device=None,
        n_steps=1000,
        return_history=False,
        solver="heun",
        clip_denoised=True,
    ):
        """Integre le probability-flow ODE par Euler ou Heun.

        Le probability-flow ODE a la meme marginale que le SDE direct, mais
        evite le bruit ajoute a chaque pas du reverse SDE. Heun est un correcteur
        d'ordre 2 qui limite l'erreur de discretisation lorsque le score varie
        fortement avec le temps.
        """
        device = self.device if device is None else device
        if n_steps < 1:
            raise ValueError("n_steps doit etre >= 1.")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver doit etre 'euler' ou 'heun'.")
        c, h, w = self.image_chw
        x = torch.randn(n_samples, c, h, w, device=device)
        history = [x.detach().cpu()] if return_history else None
        dt = -(1.0 - self.eps_time) / n_steps

        was_training = self.training
        self.eval()
        for i in range(n_steps):
            t_value = 1.0 + i * dt
            t_next_value = t_value + dt
            t = torch.full((n_samples,), t_value, device=device, dtype=x.dtype)
            beta_t = self.beta(t).reshape(-1, 1, 1, 1)
            drift = -0.5 * beta_t * x - 0.5 * beta_t * self.score(x, t)

            predictor = x + dt * drift
            if solver == "heun":
                t_next = torch.full((n_samples,), t_next_value, device=device, dtype=x.dtype)
                beta_next = self.beta(t_next).reshape(-1, 1, 1, 1)
                drift_next = -0.5 * beta_next * predictor - 0.5 * beta_next * self.score(predictor, t_next)
                x = x + 0.5 * dt * (drift + drift_next)
            else:
                x = predictor
            if return_history:
                history.append(x.detach().cpu())
        if was_training:
            self.train()
        if clip_denoised:
            x = x.clamp(-1.0, 1.0)
        return torch.stack(history) if return_history else x
