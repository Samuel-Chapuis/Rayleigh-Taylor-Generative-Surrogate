import torch


# Nombre de pas utilise pour definir un calendrier "long" de reference.
# Le seuil SNR selectionne ensuite un prefixe de ce calendrier.
SCHEDULE_REFERENCE_STEPS = 1000


def linear_beta_schedule(min_beta, max_beta, steps, device=None, dtype=torch.float32):
    """Calendrier lineaire classique des variances beta_t."""
    return torch.linspace(min_beta, max_beta, steps, device=device, dtype=dtype)


def snr_from_alpha_bar(alpha_bars):
    """Calcule SNR(t) = alpha_bar(t) / (1 - alpha_bar(t))."""
    # clamp_min evite une division par zero numerique quand alpha_bar est tres proche de 1.
    denominator = (1.0 - alpha_bars).clamp_min(torch.finfo(alpha_bars.dtype).eps)
    return alpha_bars / denominator


def diffusion_steps_from_snr(
    snr_threshold,
    min_beta=1e-4,
    max_beta=0.02,
    reference_steps=SCHEDULE_REFERENCE_STEPS,
):
    """
    Calcule le nombre de pas a conserver a partir d'un seuil SNR.

    Convention utilisee dans le notebook 1D: on construit un calendrier lineaire
    de reference, puis on garde le premier prefixe tel que
    ``SNR(t) <= snr_threshold``.
    """
    if snr_threshold is None:
        return reference_steps, max_beta
    if snr_threshold <= 0.0:
        raise ValueError(f"snr_threshold doit etre positif, recu {snr_threshold}.")
    if reference_steps < 1:
        raise ValueError(f"reference_steps doit etre >= 1, recu {reference_steps}.")

    # On travaille en float64 ici car le choix du pas d'arret depend d'un cumul
    # de produits. Cela limite les effets d'arrondi au voisinage du seuil.
    betas = linear_beta_schedule(min_beta, max_beta, reference_steps, dtype=torch.float64)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    snr = snr_from_alpha_bar(alpha_bars)
    below = torch.nonzero(snr <= snr_threshold, as_tuple=False)

    # Si le seuil n'est jamais atteint, on conserve tout le calendrier.
    if len(below) == 0:
        return reference_steps, max_beta

    # stop_index est indexe a partir de 0, alors que n_steps est un nombre de pas.
    stop_index = int(below[0].item())
    n_steps = stop_index + 1
    effective_max_beta = float(betas[stop_index].item())
    return n_steps, effective_max_beta


def get_diffusion_schedule(
    steps,
    device,
    dtype,
    min_beta=1e-4,
    max_beta=0.02,
    ddpm=None,
):
    """
    Retourne les buffers beta/alpha/alpha_bar sur le device et dtype demandes.

    Si un modele DDPM est fourni, ses buffers sont reutilises exactement. Sinon,
    un calendrier lineaire est reconstruit.
    """
    if ddpm is not None:
        # Reutiliser les buffers du modele evite une divergence subtile si le
        # modele a ete construit avec un max_beta effectif ou un autre device.
        if steps is None:
            steps = ddpm.n_steps
        if steps > ddpm.n_steps:
            raise ValueError(f"steps={steps} depasse ddpm.n_steps={ddpm.n_steps}.")

        betas = ddpm.betas[:steps].to(device=device, dtype=dtype)
        alphas = ddpm.alphas[:steps].to(device=device, dtype=dtype)
        alpha_bars = ddpm.alpha_bars[:steps].to(device=device, dtype=dtype)
        return betas, alphas, alpha_bars

    if steps is None:
        raise ValueError("steps doit etre fourni si ddpm=None.")

    betas = linear_beta_schedule(min_beta, max_beta, steps, device=device, dtype=dtype)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars
