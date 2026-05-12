import numpy as np
import pywt


def mask(vector, size):
    """
    Applique un masque centré sur un vecteur 1D.

    La fonction conserve uniquement une portion centrale du vecteur
    et met le reste à zéro.

    Args:
        vector (ndarray): vecteur d'entrée.
        size (int): taille de la zone centrale conservée.

    Returns:
        ndarray: vecteur masqué avec valeurs hors zone centrale à zéro.
    """
    v = np.zeros_like(vector)
    center = len(vector) // 2

    beg = center - size // 2
    end = center + size // 2

    v[beg:end] = vector[beg:end]
    return v


def doACP(X):
    """
    Effectue une Analyse en Composantes Principales (ACP) sur des données 2D.

    Args:
        X (ndarray): matrice de données (échantillons × variables).

    Returns:
        tuple:
            - Vt (ndarray): vecteurs propres (composantes principales)
            - S (ndarray): valeurs singulières
            - mean (ndarray): moyenne des données
    """
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt, S, X.mean(axis=0)


# ------------- #

def wavelet_lineaire(x, n, wavelet):
    """
    Compression linéaire par ondelettes en conservant les n premiers coefficients.

    Les coefficients sont gardés dans l'ordre hiérarchique des ondelettes
    (approximation puis détails), sans sélection adaptative.

    Args:
        x (ndarray): signal 1D
        n (int): nombre maximum de coefficients conservés
        wavelet (str): type d'ondelette (ex: 'db1')

    Returns:
        ndarray: signal reconstruit après compression
    """
    coeffs = pywt.wavedec(x, wavelet)

    coeffs_f = []
    coeffs_f.append(coeffs[0])  # approximation

    count = len(coeffs[0])

    for c in coeffs[1:]:
        if count + len(c) <= n:
            coeffs_f.append(c)
            count += len(c)
        else:
            coeffs_f.append(np.zeros_like(c))

    return pywt.waverec(coeffs_f, wavelet)


def wavelet_non_lineaire(x, n, wavelet):
    """
    Compression non linéaire par ondelettes (sélection des n plus grands coefficients).

    Args:
        x (ndarray): signal 1D
        n (int): nombre de coefficients conservés
        wavelet (str): type d'ondelette

    Returns:
        ndarray: signal reconstruit après compression
    """
    w = pywt.wavedec(x, wavelet)
    arr, slices = pywt.coeffs_to_array(w)

    flat = np.abs(arr).flatten()
    idx = np.argsort(flat)

    arr_f = np.zeros_like(arr)
    arr_f.flat[idx[-n:]] = arr.flat[idx[-n:]]

    coeffs_f = pywt.array_to_coeffs(arr_f, slices, output_format='wavedec')
    return pywt.waverec(coeffs_f, wavelet)


# ICI j'utilise numpy plutot que PyLab par ce que la doc insique que PyLab est deprecié par rapport
# au nouvelles version de numpy.
def fourier_lineaire(x, n):
    """
    Compression linéaire en domaine de Fourier.

    Conserve les n fréquences centrales après recentrage du spectre.

    Args:
        x (ndarray): signal 1D
        n (int): taille de la fenêtre fréquentielle conservée

    Returns:
        ndarray: signal reconstruit après filtrage fréquentiel
    """
    f = np.fft.fft(x)
    fs = np.fft.fftshift(f)

    fms = mask(fs, n)

    fm = np.fft.ifftshift(fms)
    return np.real(np.fft.ifft(fm))


def fourier_non_lineaire(x, n):
    """
    Compression non linéaire en domaine de Fourier.

    Conserve les n coefficients de plus grande amplitude.

    Args:
        x (ndarray): signal 1D
        n (int): nombre de coefficients conservés

    Returns:
        ndarray: signal reconstruit après compression fréquentielle
    """
    f = np.fft.fft(x)

    flat = np.abs(f)
    idx = np.argsort(flat)

    f_f = np.zeros_like(f)
    f_f[idx[-n:]] = f[idx[-n:]]

    return np.real(np.fft.ifft(f_f))


def acp_non_lineaire(base, x, n):
    """
    Reconstruction ACP non linéaire en conservant n composantes principales.

    Args:
        base (tuple): base ACP (Vt, S, mean)
        x (ndarray): vecteur ou signal à reconstruire
        n (int): nombre de composantes principales utilisées

    Returns:
        ndarray: reconstruction du signal
    """
    Vt, S, mean = base

    xc = x - mean

    # projection sur base ACP
    z = xc @ Vt.T

    # reconstruction avec n composantes principales
    x_rec = z[:n] @ Vt[:n]

    return x_rec + mean