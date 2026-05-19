import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import pylab as pyl
import pywt
import numpy as np
from cea.general import *


def imageplot(image, str='', sbpt=[], cmap='grey', colorbar=False, title_position="top"):
    """
    Affiche une image avec options de subplot, colormap et titre.

    Cette fonction est un wrapper de `imshow` facilitant l'affichage
    dans des grilles de sous-figures avec gestion optionnelle d'un colorbar.

    Args:
        image (ndarray): image 2D ou 3D à afficher.
        str (str, optional): titre de l'image.
        sbpt (list, optional): configuration subplot [nrows, ncols, index].
        cmap (str, optional): colormap utilisée (par défaut 'grey').
        colorbar (bool, optional): affiche une barre de couleur si True.
        title_position (str, optional): position du titre ("top" ou "left").

    Returns:
        None
    """
    if sbpt != []:
        plt.subplot(sbpt[0], sbpt[1], sbpt[2])

    ax = plt.gca()

    imgplot = ax.imshow(image, interpolation='nearest')
    imgplot.set_cmap(cmap)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(imgplot, cax=cax)

    ax.axis('off')

    if str != '':
        if title_position == "top":
            ax.set_title(str)
        elif title_position == "left":
            ax.text(
                -0.1, 0.5, str,
                transform=ax.transAxes,
                rotation=90,
                va='center',
                ha='right'
            )


def plot_sim_2d(sim, time):
    """
    Affiche une série de simulations 2D à différents instants.

    La fonction sélectionne 9 instants répartis uniformément dans le temps
    et les affiche sous forme de grille 3x3.

    Args:
        sim (ndarray): simulations de forme (n, h, w)
        time (ndarray): vecteur des temps associés

    Returns:
        None
    """
    plt.figure(figsize=(15, 15))

    for i in range(9):
        nb = i * sim.shape[0] // 9
        imageplot(
            sim[nb, :, :], #reverse(sim[nb, :, :])
            f"t={time[nb]:.2f}",
            [3, 3, i + 1],
            cmap='RdYlBu_r',
            colorbar=True,
            title_position="left"
        )


def plot_acp_modes(base, title="ACP modes", k=6):
    """
    Affiche les k premières composantes principales (ACP).

    Supporte à la fois des bases ACP 1D et 2D.

    Args:
        base (tuple): base ACP (Vt, S, mean[, shape])
        title (str, optional): titre de la figure
        k (int, optional): nombre de modes à afficher

    Returns:
        None
    """
    d = False
    if len(base) > 3:
        Vt, S, mean, (h, w) = base
        d = True
    else:
        Vt, S, mean = base

    plt.figure(figsize=(10, 6))

    for i in range(k):
        plt.subplot(2, 3, i + 1)
        plt.title(f"Mode {i+1}")

        if d:
            mode = Vt[i].reshape(h, w)
            plt.imshow(mode, cmap='gray')
            plt.axis('off')
        else:
            mode = Vt[i]
            plt.plot(mode)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_wavelet_custom_layout_2d(x, wavelet="sym6", level=3, titre=""):
    """
    Affiche la décomposition en ondelettes 2D avec layout personnalisé.

    Visualise :
    - image originale
    - approximation (LL)
    - détails (LH, HL, HH) à chaque niveau
    avec leur énergie associée.

    Args:
        x (ndarray): image 2D
        wavelet (str, optional): type d'ondelette
        level (int, optional): niveau de décomposition
        titre (str, optional): titre global de la figure

    Returns:
        None
    """

    def energy(c):
        return np.sum(c ** 2)

    coeffs = pywt.wavedec2(x, wavelet=wavelet, level=level)

    fig = plt.figure(figsize=(12, 3 * (level + 1)))

    fig.suptitle(titre, fontsize=16)
    fig.subplots_adjust(top=0.90)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    gs = gridspec.GridSpec(level + 1, 3, figure=fig)

    # ---------------- LEVEL 0 ---------------- #
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(x, cmap="gray")
    ax0.set_title("Original")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(coeffs[0], cmap="gray")
    ax1.set_title(f"LL | E={energy(coeffs[0]):.2e}")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")

    # ---------------- DETAILS ---------------- #
    for l in range(1, level + 1):

        details = coeffs[l]

        ax_lh = fig.add_subplot(gs[l, 0])
        ax_lh.imshow(details[0], cmap="gray")
        ax_lh.set_title(f"LH L{l} | E={energy(details[0]):.2e}")
        ax_lh.axis("off")

        ax_hl = fig.add_subplot(gs[l, 1])
        ax_hl.imshow(details[1], cmap="gray")
        ax_hl.set_title(f"HL L{l} | E={energy(details[1]):.2e}")
        ax_hl.axis("off")

        ax_hh = fig.add_subplot(gs[l, 2])
        ax_hh.imshow(details[2], cmap="gray")
        ax_hh.set_title(f"HH L{l} | E={energy(details[2]):.2e}")
        ax_hh.axis("off")

    plt.tight_layout()
    plt.show()