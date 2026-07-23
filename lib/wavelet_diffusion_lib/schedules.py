"""Calendrier OU continu utilisé par la diffusion wavelet."""
import math
import torch


def snr_from_time(t):
    """SNR du noyau OU : ``exp(-2t)/(1-exp(-2t))``."""
    return torch.exp(-2 * t) / torch.expm1(2 * t).clamp_min(torch.finfo(t.dtype).eps)


def time_from_snr(snr):
    """Temps terminal correspondant à une SNR prescrite."""
    return 0.5 * math.log1p(1.0 / float(snr))


def sampling_times(final_time, steps, device=None, dtype=torch.float32):
    """Temps décroissants ``T,...,0`` pour un sampler à ``steps`` évaluations."""
    if final_time <= 0 or steps < 1:
        raise ValueError("final_time doit être positif et steps >= 1")
    return torch.linspace(final_time, 0.0, int(steps) + 1, device=device, dtype=dtype)
