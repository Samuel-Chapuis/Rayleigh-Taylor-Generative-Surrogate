from pathlib import Path

import numpy as np
from scipy import linalg

from lib.PhyFID.encoder import encode_dataset, load_encoder


def feature_statistics(features):
    """
    Resume un dataset par la moyenne et covariance de ses features.
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Distance de Frechet entre deux gaussiennes de features.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Stabilisation numerique si le produit de covariance est presque singulier.
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def save_reference_statistics(images, encoder_path, stats_path, *, batch_size=128, device=None):
    """
    Sauvegarde les statistiques d'un dataset de reference.
    """
    encoder, _ = load_encoder(encoder_path, device=device)
    features = encode_dataset(images, encoder, batch_size=batch_size, device=device)
    mu, sigma = feature_statistics(features)

    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path, mu=mu, sigma=sigma)
    return mu, sigma


def load_reference_statistics(stats_path):
    stats = np.load(stats_path)
    return stats["mu"], stats["sigma"]


def compare_datasets(dataset_a, dataset_b, encoder_path, *, batch_size=128, device=None):
    """
    Compare directement deux datasets avec un encodeur deja entraine.
    """
    encoder, _ = load_encoder(encoder_path, device=device)
    features_a = encode_dataset(dataset_a, encoder, batch_size=batch_size, device=device)
    features_b = encode_dataset(dataset_b, encoder, batch_size=batch_size, device=device)
    mu_a, sigma_a = feature_statistics(features_a)
    mu_b, sigma_b = feature_statistics(features_b)
    return frechet_distance(mu_a, sigma_a, mu_b, sigma_b)


def compare_to_reference(dataset, encoder_path, stats_path, *, batch_size=128, device=None):
    """
    Compare un dataset a des statistiques de reference deja sauvegardees.
    """
    encoder, _ = load_encoder(encoder_path, device=device)
    features = encode_dataset(dataset, encoder, batch_size=batch_size, device=device)
    mu, sigma = feature_statistics(features)
    ref_mu, ref_sigma = load_reference_statistics(stats_path)
    return frechet_distance(ref_mu, ref_sigma, mu, sigma)
