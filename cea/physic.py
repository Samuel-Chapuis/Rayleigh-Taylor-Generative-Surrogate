import numpy as np

def taille_zone_melange(image2d, epsilon=0.1, marge=10, list=False):
    """
    Calcule les bornes inférieure et supérieure de la zone de mélange.

    Args:
        image2d (numpy array): image de simulation 2D de forme (y, x)
        epsilon (float): seuil pour définir la zone de mélange
        marge (int): marge en pixels ajoutée au-dessus et en dessous
        list (bool): si True, retourne les listes des bornes en plus lieu des valeurs uniques
    Returns:
        tuple: (borne_inf, borne_sup, borne_inf_col, borne_sup_col)
    """
    if image2d.ndim != 2:
        raise ValueError(f"taille_zone_melange attend une image 2D, reçu {image2d.shape}")

    borne_inf_col = np.full(image2d.shape[1], np.nan, dtype=float)
    borne_sup_col = np.full(image2d.shape[1], np.nan, dtype=float)

    seuil_bas = 1 - epsilon
    seuil_haut = epsilon

    for col in range(image2d.shape[1]):
        column = image2d[:, col]

        for row in range(image2d.shape[0]):
            if column[row] < seuil_bas:
                borne_inf_col[col] = row
                break

        for row in range(image2d.shape[0] - 1, -1, -1):
            if column[row] > seuil_haut:
                borne_sup_col[col] = row
                break

    valid_inf = borne_inf_col[~np.isnan(borne_inf_col)]
    valid_sup = borne_sup_col[~np.isnan(borne_sup_col)]

    if valid_inf.size == 0 or valid_sup.size == 0:
        return None, None, borne_inf_col, borne_sup_col

    borne_inf = max(0, int(np.min(valid_inf)) - marge)
    borne_sup = min(image2d.shape[0] - 1, int(np.max(valid_sup)) + marge)

    if list:
        return borne_inf, borne_sup, borne_inf_col, borne_sup_col
    else:
        return borne_inf, borne_sup