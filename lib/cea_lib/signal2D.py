import numpy as np
import pywt
import cv2
try:
    import torch
except ImportError:
    torch = None


def normalize2d(matrix, value_min=0, value_max=1):
    """
    Normalise une matrice 2D ou un dataset d'images 2D.

    Pour un dataset de forme (N, H, W) ou (N, C, H, W), chaque image est
    normalisee independamment sur ses deux dernieres dimensions.

    Args:
        matrix (ndarray | torch.Tensor): Matrice 2D ou dataset d'entree.
        value_min (float): Valeur minimale de l'échelle.
        value_max (float): Valeur maximale de l'échelle.


    Returns:
        ndarray | torch.Tensor: Donnees normalisees, avec le meme type de conteneur.
    """
    if torch is not None and torch.is_tensor(matrix):
        data = matrix if matrix.is_floating_point() else matrix.float()
        reduce_dims = tuple(range(data.ndim)) if data.ndim <= 2 else (-2, -1)

        min_val = data.amin(dim=reduce_dims, keepdim=True)
        max_val = data.amax(dim=reduce_dims, keepdim=True)
        scale = max_val - min_val
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        normalized = (data - min_val) / safe_scale * (value_max - value_min) + value_min
        return torch.where(scale == 0, torch.zeros_like(data), normalized)

    data = np.asarray(matrix)
    reduce_dims = None if data.ndim <= 2 else (-2, -1)

    min_val = np.min(data, axis=reduce_dims, keepdims=True)
    max_val = np.max(data, axis=reduce_dims, keepdims=True)
    scale = max_val - min_val
    safe_scale = np.where(scale == 0, 1, scale)

    if np.all(scale == 0):
        return np.zeros_like(data)

    normalized = (data - min_val) / safe_scale * (value_max - value_min) + value_min
    return np.where(scale == 0, np.zeros_like(data), normalized)


def mask2d(matrix, sizex, sizey=None, crop=False):
    """
    Applique un masque carré centré sur une matrice 2D.

    Cette fonction extrait ou conserve une région centrale de la matrice
    en mettant à zéro le reste, selon le paramètre `crop`.

    Args:
        matrix (ndarray): Matrice 2D d'entrée.
        sizex (int): Largeur du masque.
        sizey (int, optional): Hauteur du masque. Si None, égal à sizex.
        crop (bool, optional): Si True, retourne uniquement la région masquée.
            Sinon, conserve la taille originale avec les valeurs hors masque à 0.

    Returns:
        ndarray: Matrice masquée ou sous-matrice découpée.
    """
    if sizey is None:
        sizey = sizex

    h, w = matrix.shape
    ch, cw = h // 2, w // 2

    bh = ch - sizey // 2
    eh = ch + sizey // 2

    bw = cw - sizex // 2
    ew = cw + sizex // 2

    if crop:
        return matrix[bh:eh, bw:ew]
    else:
        v = np.zeros_like(matrix)
        v[bh:eh, bw:ew] = matrix[bh:eh, bw:ew]
        return v


def resize2d(matrix, x2, y2, interpolation="linear"):
    """
    Redimensionne une image/matrice 2D.

    Args:
        matrix (ndarray): image d'entrée
        x2 (int): largeur cible
        y2 (int): hauteur cible
        interpolation (str): méthode d'interpolation
            ("nearest", "linear", "cubic", "area")

    Returns:
        ndarray: image redimensionnée
    """
    interp_dict = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA
    }

    interp = interp_dict.get(interpolation, cv2.INTER_LINEAR)

    return cv2.resize(matrix, (x2, y2), interpolation=interp)


def doACP_2d(X):
    """
    Effectue une Analyse en Composantes Principales (ACP) sur des images 2D.

    Args:
        X (ndarray): Données de forme (n_samples, height, width).

    Returns:
        tuple:
            - Vt (ndarray): vecteurs propres (composantes principales)
            - S (ndarray): valeurs singulières
            - mean (ndarray): moyenne des données vectorisées
            - shape (tuple): forme (height, width) des images originales
    """
    n, h, w = X.shape
    X_flat = X.reshape(n, h * w)

    Xc = X_flat - X_flat.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    mean = X_flat.mean(axis=0)
    return Vt, S, mean, (h, w)


def acp_non_lineaire_2d(base, x, n):
    """
    Reconstruction ACP non linéaire tronquée à n composantes.

    Args:
        base (tuple): résultat de doACP_2d (Vt, S, mean, shape)
        x (ndarray): image à reconstruire
        n (int): nombre de composantes principales utilisées

    Returns:
        ndarray: image reconstruite
    """
    Vt, S, mean, shape = base
    h, w = shape

    xc = x.flatten() - mean
    z = xc @ Vt.T

    x_rec = z[:n] @ Vt[:n]
    return (x_rec + mean).reshape(h, w)


def wavelet_lineaire_2d(x, n, wavelet):
    """
    Compression linéaire par ondelettes en gardant n coefficients.

    Args:
        x (ndarray): image 2D
        n (int): nombre maximal de coefficients conservés
        wavelet (str): type d'ondelette (ex: 'db1')

    Returns:
        ndarray: image reconstruite après compression
    """
    coeffs = pywt.wavedec2(x, wavelet)

    coeffs_f = [coeffs[0]]
    count = coeffs[0].size

    for detail in coeffs[1:]:
        new_detail = []
        for c in detail:
            if count + c.size <= n:
                new_detail.append(c)
                count += c.size
            else:
                new_detail.append(np.zeros_like(c))
        coeffs_f.append(tuple(new_detail))

    return pywt.waverec2(coeffs_f, wavelet)


def wavelet_non_lineaire_2d(x, n, wavelet, return_coeffs=False):
    """
    Compression non linéaire par ondelettes (sélection des n plus grands coefficients).

    Args:
        x (ndarray): image 2D
        n (int): nombre de coefficients conservés
        wavelet (str): type d'ondelette
        return_coeffs (bool): si True, retourne aussi les coefficients

    Returns:
        ndarray or tuple:
            - image reconstruite
            - (optionnel) coefficients et structures internes
    """
    coeffs = pywt.wavedec2(x, wavelet)
    arr, slices = pywt.coeffs_to_array(coeffs)

    flat = np.abs(arr).ravel()
    idx = np.argsort(flat)

    arr_f = np.zeros_like(arr)
    arr_f.flat[idx[-n:]] = arr.flat[idx[-n:]]

    coeffs_f = pywt.array_to_coeffs(arr_f, slices, output_format='wavedec2')
    x_rec = pywt.waverec2(coeffs_f, wavelet)

    if return_coeffs:
        return x_rec, coeffs_f, slices, arr_f
    else:
        return x_rec


def square_mask(matrix, center, size):
    """
    Applique un masque carré centré sur une matrice.

    Args:
        matrix (ndarray): matrice 2D
        center (int): centre du masque
        size (int): taille du carré

    Returns:
        ndarray: matrice masquée
    """
    m0 = np.zeros_like(matrix)
    beg = center - size // 2
    end = center + size // 2

    m0[beg:end, beg:end] = matrix[beg:end, beg:end]
    return m0


def fourier_lineaire_2d(x, n):
    """
    Reconstruction par Fourier en conservant les basses fréquences.

    Args:
        x (ndarray): image 2D
        n (int): nombre de coefficients (approximation carrée)

    Returns:
        ndarray: image reconstruite
    """
    f = np.fft.fft2(x, norm="ortho")
    fs = np.fft.fftshift(f)

    q = int(np.sqrt(n))
    center = x.shape[0] // 2

    fms = square_mask(fs, center, q)

    fm = np.fft.ifftshift(fms)

    return np.real(np.fft.ifft2(fm, norm="ortho"))


def fourier_non_lineaire_2d(x, n):
    """
    Reconstruction Fourier en conservant les n plus grandes amplitudes.

    Args:
        x (ndarray): image 2D
        n (int): nombre de coefficients conservés

    Returns:
        ndarray: image reconstruite
    """
    f = np.fft.fft2(x)

    flat = np.abs(f).ravel()
    idx = np.argsort(flat)

    f_f = np.zeros_like(f)
    f_f.flat[idx[-n:]] = f.flat[idx[-n:]]

    return np.real(np.fft.ifft2(f_f))
