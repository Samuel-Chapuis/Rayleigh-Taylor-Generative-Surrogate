import h5py
import numpy as np
from collections import defaultdict
from cea.signal2D import *


def load_RTCEA(file):
    """
    Charge un fichier de simulation au format RTCEA (HDF5).

    Args:
        file (str): chemin du fichier HDF5

    Returns:
        tuple:
            - data (ndarray): données des simulations
            - labels (ndarray): labels associés
    """
    d = h5py.File(file, "r")
    extr_data = d["data"]
    extr_label = d["labels"]

    data = np.array(extr_data)
    labels = np.array(extr_label)

    return data, labels


def create_atwood_dic(labels):
    """
    Crée un dictionnaire indexant les simulations par nombre d'Atwood.

    Args:
        labels (ndarray): tableau de labels

    Returns:
        dict: clés = Atwood (float arrondi), valeurs = indices des simulations
    """
    atwood_list = np.unique(labels[:, 0])
    atwood_idx = defaultdict(list)

    for i in range(labels.shape[0]):
        key = round(float(labels[i, 0]), 3)
        atwood_idx[key].append(i)

    return atwood_idx


def data_normalise_2d(data, labels, marge=0, square=False):
    """
    Normalise et recadre des simulations 2D pour un nombre d'Atwood donné.

    La fonction :
    - sélectionne les simulations correspondant à un Atwood
    - normalise un paramètre de longueur
    - applique un masquage spatial
    - redimensionne les images

    Args:
        data (ndarray): simulations 3D (n, h, w)
        labels (ndarray): labels associés
        atwood (float): valeur du nombre d'Atwood

    Returns:
        tuple:
            - sim_norm (ndarray): simulations normalisées
            - t (ndarray): temps associé aux simulations
    """

    t = labels[:, 1]
    L = labels[:, 2]
    L_norm = (L - L.min()) / (L.max() - L.min())

    sim = data[:, :, :]

    if square:
        # Choisis la taille carrée cible (par exemple, la plus grande dimension)
        target_size = max(data.shape[1], data.shape[2])
        sim_norm = np.zeros((data.shape[0], target_size, target_size))
    else:
        sim_norm = np.zeros_like(data)

    for i in range(data.shape[0]):
        simi = data[i, :, :]

        cropped = mask2d(
            simi,
            data.shape[1] * 2,
            int(data.shape[2] * L_norm[i] + marge * data.shape[2]),
            True
        )

        if square:
            # Redimensionne en carré
            sim_norm[i] = resize2d(cropped, target_size, target_size)
        else:
            sim_norm[i] = resize2d(cropped, simi.shape[1], simi.shape[0])


    return sim_norm