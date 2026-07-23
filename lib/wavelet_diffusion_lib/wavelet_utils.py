from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .ConditionalDDPM import WaveletConditionalDDPM
from .UNet import UNet
from .schedules import snr_from_time


def load_wave_tensor(path, expected_channels=4):
    """Charge un tenseur wavelet deja pretraite au format (N, C, H, W)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset wavelet introuvable: {path}. Lance wave_data_mining_CEA.py "
            f"pour generer les fichiers j*_training.pt / validation.pt / test.pt."
        )

    data = torch.load(path, map_location="cpu").float()
    if data.ndim != 4:
        raise ValueError(f"Dataset attendu en (N, C, H, W), recu {tuple(data.shape)}.")
    if data.shape[1] != expected_channels:
        raise ValueError(f"Dataset attendu avec {expected_channels} canaux, recu {data.shape[1]}.")
    return data


def channel_stats(data):
    """Statistiques canal par canal, partagees par train/validation/test."""
    mean = data.mean(dim=(0, 2, 3))
    std = data.std(dim=(0, 2, 3)).clamp_min(1e-6)
    return mean, std


def normalize_with_stats(data, mean, std):
    """Normalisation affine avec des statistiques de forme (C,)."""
    return (data - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)


def make_loader(data, batch_size, shuffle):
    """Construit un DataLoader sans label: chaque batch contient seulement les coefficients."""
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def show_wave_channels(viz, coeffs, title):
    """
    Sauvegarde une grille simple: chaque coefficient devient une image affichee.
    """
    n = min(8, len(coeffs))
    images = coeffs[:n].reshape(n * coeffs.shape[1], 1, coeffs.shape[2], coeffs.shape[3])
    viz.show_images(images, title)


def build_wavelet_model(config, image_hw, coeff_mean, coeff_std):
    """Construit le U-Net et l'encapsule dans le DDPM conditionnel wavelet."""
    # Le U-Net voit le prior cA concatene aux details bruites, mais ne predit
    # que le bruit ajoute aux canaux de details.
    network = UNet(
        time_emb_dim=config.time_emb_dim,
        size=image_hw[0],
        in_channels=config.input_channels,
        out_channels=config.target_channels,
        depth=config.unet_depth,
        blocks_per_level=config.unet_blocks_per_level,
        base_channels=config.unet_base_channels,
        norm_type=getattr(config, "unet_norm_type", "group"),
        norm_max_groups=getattr(config, "unet_norm_max_groups", 8),
    )
    return WaveletConditionalDDPM(
        network,
        final_time=getattr(config, "final_time", 5.0),
        snr_terminal=getattr(config, "snr_terminal", None),
        sampling_steps=getattr(config, "sampling_steps", 16),
        sampling_eta=getattr(config, "sampling_eta", 1.0),
        device=config.device,
        prior_channels=config.prior_channels,
        target_channels=config.target_channels,
        image_hw=image_hw,
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
    ).to(config.device)


def do_diffusion_until_snr(
    data,
    snr_threshold=None,
    steps=16,
    ddpm=None,
    min_beta=None,
    max_beta=None,
    reference_steps=None,
    prior_channels=1,
    diffuse_prior=False,
    keep_every=1,
    detach_to_cpu=True,
):
    """
    Applique le processus direct de diffusion jusqu'a un seuil de SNR.

    Par defaut, le prior basse frequence cA est conserve et seuls les canaux de
    detail cH/cV/cD sont bruites. Utiliser ``diffuse_prior=True`` pour bruiter
    tous les canaux.

    ``reference_steps`` controle le calendrier long sur lequel le seuil SNR est
    evalue lorsque ``steps`` n'est pas fourni.
    """
    if data.ndim != 4:
        raise ValueError(f"data attendu en (N, C, H, W), recu {tuple(data.shape)}.")
    if keep_every < 1:
        raise ValueError(f"keep_every doit etre >= 1, recu {keep_every}.")

    # Fonction de visualisation historique : le processus utilisé par la
    # bibliothèque est désormais OU continu, sans calendrier beta discret.
    data = data.to(
        device=ddpm.device if ddpm is not None and ddpm.device is not None else data.device,
    )

    # En mode conditionnel, les premiers canaux sont le prior non diffuse.
    # Pour les coefficients wavelet j=1 usuels: cA est conserve, cH/cV/cD diffusent.
    if diffuse_prior:
        prior = None
        xt = data
    else:
        if prior_channels < 0 or prior_channels >= data.shape[1]:
            raise ValueError(
                f"prior_channels doit etre dans [0, {data.shape[1] - 1}], recu {prior_channels}."
            )
        prior = data[:, :prior_channels]
        xt = data[:, prior_channels:]

    final_time = ddpm.final_time if ddpm is not None else 5.0
    if snr_threshold is not None:
        final_time = 0.5 * torch.log1p(torch.tensor(1.0 / snr_threshold)).item()
    n_steps = steps or (ddpm.sampling_steps if ddpm is not None else 16)
    times = torch.linspace(0.0, final_time, n_steps + 1, device=data.device, dtype=data.dtype)[1:]

    # Les echantillons sauvegardes gardent la meme convention de canaux que
    # l'entree: prior en premier, details ensuite.
    def pack_sample(details):
        sample = details if prior is None else torch.cat((prior, details), dim=1)
        sample = sample.detach()
        return sample.cpu() if detach_to_cpu else sample.clone()

    samples = [pack_sample(xt)]
    last_step = 0
    x0 = xt

    with torch.no_grad():
        for i, t in enumerate(times):
            alpha = torch.exp(-2 * t)
            xt = alpha.sqrt() * x0 + (1-alpha).sqrt() * torch.randn_like(x0)
            last_step = i + 1

            if last_step % keep_every == 0:
                samples.append(pack_sample(xt))

            if snr_threshold is not None:
                snr = snr_from_time(t)
                # Meme convention que le notebook: arret des que SNR <= seuil.
                if float(snr.item()) <= snr_threshold:
                    if last_step % keep_every != 0:
                        samples.append(pack_sample(xt))
                    break

    return samples, last_step
