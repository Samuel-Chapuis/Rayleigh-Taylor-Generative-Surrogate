"""Conditional VP-SDE model for wavelet detail coefficients."""

import torch
import torch.nn as nn


class WaveletConditionalSGM(nn.Module):
    """VP-SDE on wavelet details conditioned by the approximation channel.

    The network receives ``[prior, noisy_details]`` and predicts either epsilon
    or the bounded v-parameterization. The prior is never diffused.
    """

    def __init__(
        self,
        network,
        beta_min=0.1,
        beta_max=20.0,
        device=None,
        prior_channels=1,
        target_channels=3,
        image_hw=(32, 32),
        coeff_mean=None,
        coeff_std=None,
        eps_time=1e-2,
        prediction_type="v",
    ):
        super().__init__()
        if beta_min <= 0 or beta_max < beta_min:
            raise ValueError("Il faut 0 < beta_min <= beta_max.")
        if not 0 < eps_time < 1:
            raise ValueError("eps_time doit appartenir a (0, 1).")
        if prediction_type not in {"epsilon", "v"}:
            raise ValueError("prediction_type doit etre 'epsilon' ou 'v'.")

        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.eps_time = float(eps_time)
        self.prediction_type = prediction_type
        self.prior_channels = int(prior_channels)
        self.target_channels = int(target_channels)
        self.image_hw = tuple(image_hw)
        self.device = device
        self.network = network.to(device)

        n_channels = self.prior_channels + self.target_channels
        if coeff_mean is None:
            coeff_mean = torch.zeros(n_channels)
        if coeff_std is None:
            coeff_std = torch.ones(n_channels)
        self.register_buffer("coeff_mean", coeff_mean.float().reshape(1, -1, 1, 1))
        self.register_buffer("coeff_std", coeff_std.float().reshape(1, -1, 1, 1))

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def integrated_beta(self, t):
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t.square()

    def alpha_bar(self, t):
        return torch.exp(-self.integrated_beta(t))

    def marginal_std(self, t):
        return torch.sqrt((1.0 - self.alpha_bar(t)).clamp_min(1e-12))

    def forward(self, details_0, t, eps=None):
        """Sample the exact VP marginal ``q(details_t | details_0)``."""
        t = t.to(device=details_0.device, dtype=details_0.dtype).reshape(-1)
        if t.shape[0] != details_0.shape[0]:
            raise ValueError("Il faut un temps par element du batch.")
        if eps is None:
            eps = torch.randn_like(details_0)
        alpha = self.alpha_bar(t).clamp_min(1e-12).sqrt()
        sigma = self.marginal_std(t)
        return alpha[:, None, None, None] * details_0 + sigma[:, None, None, None] * eps

    def prediction_target(self, details_0, t, eps):
        if self.prediction_type == "epsilon":
            return eps
        alpha = self.alpha_bar(t).clamp_min(1e-12).sqrt()
        sigma = self.marginal_std(t)
        return alpha[:, None, None, None] * eps - sigma[:, None, None, None] * details_0

    def predict_epsilon(self, noisy_details, t, prior):
        t = t.to(device=noisy_details.device, dtype=noisy_details.dtype).reshape(-1)
        prediction = self.network(torch.cat((prior, noisy_details), dim=1), t)
        if self.prediction_type == "epsilon":
            return prediction
        alpha = self.alpha_bar(t).clamp_min(1e-12).sqrt()
        sigma = self.marginal_std(t)
        return sigma[:, None, None, None] * noisy_details + alpha[:, None, None, None] * prediction

    def score(self, noisy_details, t, prior):
        sigma = self.marginal_std(t).reshape(-1, 1, 1, 1)
        return -self.predict_epsilon(noisy_details, t, prior) / sigma

    def denormalize_coeffs(self, coeffs):
        return coeffs * self.coeff_std + self.coeff_mean

    def normalize_coeffs(self, coeffs):
        return (coeffs - self.coeff_mean) / self.coeff_std

    @torch.no_grad()
    def sample(
        self,
        prior,
        device=None,
        n_steps=256,
        solver="heun",
        return_history=False,
    ):
        """Generate details with the conditional probability-flow ODE."""
        if n_steps < 1:
            raise ValueError("n_steps doit etre >= 1.")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver doit etre 'euler' ou 'heun'.")
        device = self.device if device is None else device
        prior = prior.to(device)
        n, channels, h, w = prior.shape
        if channels != self.prior_channels:
            raise ValueError(f"Prior attendu avec {self.prior_channels} canaux, recu {channels}.")

        x = torch.randn(n, self.target_channels, h, w, device=device, dtype=prior.dtype)
        history = [torch.cat((prior, x), dim=1).cpu()] if return_history else None
        dt = -(1.0 - self.eps_time) / n_steps
        was_training = self.training
        self.eval()

        def drift(state, time):
            beta = self.beta(time).reshape(-1, 1, 1, 1)
            return -0.5 * beta * state - 0.5 * beta * self.score(state, time, prior)

        for index in range(n_steps):
            t_value = 1.0 + index * dt
            t_next_value = t_value + dt
            t = torch.full((n,), t_value, device=device, dtype=x.dtype)
            current_drift = drift(x, t)
            predictor = x + dt * current_drift
            if solver == "heun":
                t_next = torch.full((n,), t_next_value, device=device, dtype=x.dtype)
                x = x + 0.5 * dt * (current_drift + drift(predictor, t_next))
            else:
                x = predictor
            if not torch.isfinite(x).all():
                raise FloatingPointError("La trajectoire SGM wavelet contient des valeurs non finies.")
            if return_history:
                history.append(torch.cat((prior, x), dim=1).cpu())

        if was_training:
            self.train()
        return torch.stack(history) if return_history else torch.cat((prior, x), dim=1)
