import os
import sys
from pathlib import Path
import csv

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

pytorch_fid_src = project_root / "lib" / "pytorch-fid-master" / "src"
if str(pytorch_fid_src) not in sys.path:
    sys.path.insert(0, str(pytorch_fid_src))

from pytorch_fid.fid_score import calculate_frechet_distance as pytorch_fid_distance
from pytorch_fid.inception import InceptionV3

from lib.PhyFID.metrics import compare_datasets, compare_to_reference
from lib.PhyFID.utils import evaluate_phyfid_dataset, load_pt_dataset
from lib.cea_lib import*

## Parametres ##

DATA_ROOT = project_root / "data" / "RT28"
PROCESSED_ROOT = DATA_ROOT / "processed"
VAL_PT = os.path.join(PROCESSED_ROOT, "validation.pt")

GENERATED_PT = project_root / "generated" / "generated_dataset.pt"

PHYFID_ENCODER = project_root / "outputs" / "phyFID" / "phyfid_encoder.pt"
PHYFID_STATS = project_root / "outputs" / "phyFID" / "phyfid_val_stats.npz"
PHYFID_BATCH_SIZE = 128
ERROR_CSV = project_root / "outputs" / "error_metrics.csv"

## Chargement des datasets ##

def normalize_generated_dataset(images, value_min=0, value_max=255):
    """Remet les sorties DDPM dans la convention uint8-like [0, 255] du dataset RT.

    La normalisation est faite image par image sur les deux dimensions spatiales,
    comme lors du preprocessing des champs RT.
    """
    data = images.float() if torch.is_tensor(images) else torch.as_tensor(images).float()
    reduce_dims = tuple(range(data.ndim)) if data.ndim <= 2 else (-2, -1)
    min_value = data.amin(dim=reduce_dims, keepdim=True)
    max_value = data.amax(dim=reduce_dims, keepdim=True)
    scale = max_value - min_value
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    normalized = (data - min_value) / safe_scale
    normalized = normalized * (value_max - value_min) + value_min
    return torch.where(scale == 0, torch.zeros_like(data), normalized)


def to_txy(images):
    """
    Convertit un dataset d'images en forme (N, H, W).
    """
    images = torch.as_tensor(images)
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    if images.ndim != 3:
        raise ValueError(f"Images attendues en (N, H, W) ou (N, 1, H, W), recu {tuple(images.shape)}")
    return images


def normalize_01(tensor, normalize=True):
    """Normalise les champs dans la convention ``[0, 1]``.

    Args:
        tensor (torch.Tensor | np.ndarray): Donnees de forme ``(t, x, y)`` ou
            ``(t, 1, x, y)``.
        normalize (bool, optional): Si ``True``, divise par 255 lorsque les
            valeurs semblent etre dans la convention ``[0, 255]``. Defaults to True.

    Returns:
        torch.Tensor: Donnees flottantes de forme ``(t, x, y)``.
    """
    tensor = to_txy(tensor)
    if normalize and tensor.max() > 1.5:
        tensor = tensor / 255.0
    return tensor


def as_txy_numpy(data):
    """Convertit des champs images en ``ndarray`` de forme ``(t, x, y)``.

    Args:
        data (torch.Tensor | np.ndarray): Donnees de forme ``(t, x, y)`` ou
            ``(t, 1, x, y)``.

    Returns:
        np.ndarray: Donnees flottantes de forme ``(t, x, y)``.

    Raises:
        ValueError: Si la forme des donnees n'est pas supportee.
    """
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.asarray(data, dtype=float)
    if data.ndim == 4 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 3:
        raise ValueError(f"Donnees attendues en (t, x, y) ou (t, 1, x, y), recu {data.shape}")
    return data


def transformation_Fluctuation_Eenergy_Spectrum(data, ax):
    """Calcule le spectre d'energie des fluctuations d'un champ.

    Args:
        data (torch.Tensor | np.ndarray): Champ de forme ``(t, x, y)`` ou
            ``(t, 1, x, y)``.
        ax (int): Axe de Fourier. ``-1`` applique une FFT 2D sur ``(x, y)`` ;
            toute autre valeur applique une FFT 1D sur cet axe.

    Returns:
        np.ndarray: Spectre d'energie ``|FFT(delta rho)|^2`` de forme compatible
            avec l'entree convertie en ``(t, x, y)``.
    """
    data = as_txy_numpy(data)
    fluctuation_val = np.abs(data - data.mean())

    if ax == -1:
        fluctuation_spectrum_val = np.fft.fftshift(np.fft.fft2(fluctuation_val, axes=(-2, -1)), axes=(-2, -1))
    else:
        fluctuation_spectrum_val = np.fft.fftshift(
            np.fft.fft(fluctuation_val, axis=ax),
            axes=ax,
        )

    energy_spectrum_val = np.abs(fluctuation_spectrum_val) ** 2
    return energy_spectrum_val


def line_metric(validation, generated, metric_name="mu"):
    """Compare deux jeux de champs par profils de lignes.

    Args:
        validation (torch.Tensor | np.ndarray): Jeu de reference de forme
            ``(t, x, y)`` ou ``(t, 1, x, y)``.
        generated (torch.Tensor | np.ndarray): Jeu compare de forme
            ``(t, x, y)`` ou ``(t, 1, x, y)``.
        metric_name (str, optional): Type de profil compare. ``"mu"`` compare
            la moyenne temporelle de ``mean_y`` ; ``"sigma"`` compare son
            ecart-type temporel. Defaults to "mu".

    Returns:
        tuple[float, np.ndarray, np.ndarray]: Distance normalisee, profil du
            jeu compare et profil du jeu de validation.

    Raises:
        ValueError: Si les deux jeux n'ont pas le meme nombre de lignes ou si
            ``metric_name`` n'est pas supporte.
    """
    validation = as_txy_numpy(validation)
    generated = as_txy_numpy(generated)
    if validation.shape[1] != generated.shape[1]:
        raise ValueError(f"Le nombre de lignes x doit etre le meme pour validation et generated: {validation.shape[1]} vs {generated.shape[1]}")

    n_x = validation.shape[1]
    mu_val = validation.mean(axis=2)
    mu_gen = generated.mean(axis=2)
    if metric_name == "mu":
        MU_l_val = np.mean(mu_val, axis=0)
        MU_l_gen = np.mean(mu_gen, axis=0)
        distance = np.linalg.norm(MU_l_val - MU_l_gen) / n_x
        return distance, MU_l_gen, MU_l_val 
    elif metric_name == "sigma":
        SIGMA_l_val = np.std(mu_val, axis=0)
        SIGMA_l_gen = np.std(mu_gen, axis=0)
        distance = np.linalg.norm(SIGMA_l_val - SIGMA_l_gen) / n_x
        return distance, SIGMA_l_gen, SIGMA_l_val
    else:
        raise ValueError(f"Metric non supportee: {metric_name}")


def line_errors_to_list(values):
    """
    Convertit un vecteur d'erreurs par ligne en liste plate pour csv.writerow.
    """
    return np.asarray(values, dtype=float).ravel().tolist()



def main():
    val = load_pt_dataset(VAL_PT)
    generated_raw = load_pt_dataset(GENERATED_PT)
    generated_dataset = normalize_generated_dataset(generated_raw)


    ## phyFID ##
    PHYFID_N_IMAGES = min(1000, len(val) // 2, len(generated_dataset))
    phyfid_reference = val[:PHYFID_N_IMAGES]

    phyFIDscore = compare_datasets(
        phyfid_reference,
        generated_dataset[:PHYFID_N_IMAGES],
        encoder_path=PHYFID_ENCODER,
        batch_size=PHYFID_BATCH_SIZE,
    )

    print(f"PhyFID score (val vs generated): {phyFIDscore:.4f}")


    ## Profils de lignes ##
    val_stats = normalize_01(val)
    generated_stats = normalize_01(generated_dataset)
    mu_l_dist_gen, mu_l_gen, mu_l_val = line_metric(val_stats, generated_stats, metric_name="mu")

    mu_table = [mu_l_dist_gen, np.abs(mu_l_val - mu_l_gen) / mu_l_val.size]
    print(mu_table)

    ## Spectre d'energie des fluctuations ##
    Smu_l_dist_gen, Smu_l_gen, Smu_l_val = line_metric(transformation_Fluctuation_Eenergy_Spectrum(val_stats, 1), 
                                                   transformation_Fluctuation_Eenergy_Spectrum(generated_stats, 1), 
                                                   metric_name="mu")
    
    Smu_table = [Smu_l_dist_gen, np.abs(Smu_l_val - Smu_l_gen) / Smu_l_val.size]
    print(Smu_table)


    csv1stLine = ['Métriques', 'Moyennes']
    for i in range(len(mu_l_gen)):
        csv1stLine.append(f'l{i+1}')
    ##Sauvegarde des resultats dans un fichier CSV##
    with open(ERROR_CSV, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv1stLine)
        writer.writerow(['PhyFID', phyFIDscore])
        writer.writerow([
            'Mean Line Profile (mu)',
            mu_l_dist_gen,
            *line_errors_to_list(np.abs(mu_l_val - mu_l_gen)),
        ])
        writer.writerow([
            'Fluctuation Energy Spectrum (mu)',
            Smu_l_dist_gen,
            *line_errors_to_list(np.abs(Smu_l_val - Smu_l_gen)),
        ])

if __name__ == "__main__":
    main()
