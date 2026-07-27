"""Utilitaires du calendrier OU continu."""
import math
import torch


def snr_from_time(t):
    return 1.0 / torch.expm1(2 * t).clamp_min(torch.finfo(t.dtype).eps)


def time_from_snr(snr):
    return 0.5 * math.log1p(1.0 / float(snr))


def sampling_times(final_time, steps, device=None, dtype=torch.float32):
    if final_time <= 0 or steps < 1:
        raise ValueError("final_time doit être positif et steps >= 1")
    return torch.linspace(final_time, 0, int(steps) + 1, device=device, dtype=dtype)
