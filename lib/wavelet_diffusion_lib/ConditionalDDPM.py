import math

import torch
import torch.nn as nn


class WaveletConditionalDDPM(nn.Module):
    """Diffusion OU continue des détails conditionnée par cA."""

    def __init__(
        self, network, final_time=5.0, sampling_steps=16, snr_terminal=None,
        sampling_eta=1.0, device=None, prior_channels=1, target_channels=3,
        image_hw=(32, 32), coeff_mean=None, coeff_std=None,
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
        self.prior_channels = prior_channels
        self.target_channels = target_channels
        self.image_hw = image_hw
        self.network = network.to(device)
        if coeff_mean is None:
            coeff_mean = torch.zeros(prior_channels + target_channels)
        if coeff_std is None:
            coeff_std = torch.ones(prior_channels + target_channels)
        self.register_buffer("coeff_mean", coeff_mean.float().reshape(1, -1, 1, 1))
        self.register_buffer("coeff_std", coeff_std.float().reshape(1, -1, 1, 1))

    def forward(self, details_0, t, eta=None):
        if details_0.shape[1] != self.target_channels:
            raise ValueError(f"Details attendus avec {self.target_channels} canaux.")
        t = self._flatten_time(t, details_0.device, details_0.dtype)
        if eta is None:
            eta = torch.randn_like(details_0)
        alpha = torch.exp(-2.0 * t).reshape(-1, 1, 1, 1)
        return alpha.sqrt() * details_0 + (1 - alpha).clamp_min(0).sqrt() * eta

    def backward(self, noisy_details, t, prior):
        t = self._flatten_time(t, noisy_details.device, noisy_details.dtype)
        return self.network(torch.cat((prior, noisy_details), dim=1), t)

    def sample(self, prior, device=None):
        device = device or self.device
        prior = prior.to(device)
        n, c, h, w = prior.shape
        if c != self.prior_channels:
            raise ValueError(f"Prior attendu avec {self.prior_channels} canaux.")
        x = torch.randn(n, self.target_channels, h, w, device=device)
        times = torch.linspace(self.final_time, 0.0, self.sampling_steps + 1,
                               device=device, dtype=x.dtype)
        with torch.no_grad():
            for i in range(self.sampling_steps):
                t, t_prev = times[i], times[i + 1]
                t_batch = torch.full((n,), t, device=device, dtype=x.dtype)
                eps = self.backward(x, t_batch, prior)
                alpha_t, alpha_prev = torch.exp(-2*t), torch.exp(-2*t_prev)
                x0_hat = (x - (1-alpha_t).sqrt() * eps) / alpha_t.sqrt()
                ratio = (alpha_t / alpha_prev).clamp(0, 1)
                sigma = self.sampling_eta * torch.sqrt(
                    ((1-alpha_prev)/(1-alpha_t).clamp_min(1e-12)) * (1-ratio)
                )
                residual = (1-alpha_prev-sigma.square()).clamp_min(0).sqrt()
                x = alpha_prev.sqrt()*x0_hat + residual*eps
                if i < self.sampling_steps - 1:
                    x = x + sigma * torch.randn_like(x)
        return torch.cat((prior, x), dim=1)

    def denormalize_coeffs(self, coeffs):
        return coeffs * self.coeff_std + self.coeff_mean

    def normalize_coeffs(self, coeffs):
        return (coeffs - self.coeff_mean) / self.coeff_std

    @staticmethod
    def _flatten_time(t, device, dtype):
        return torch.as_tensor(t, device=device, dtype=dtype).reshape(-1)
