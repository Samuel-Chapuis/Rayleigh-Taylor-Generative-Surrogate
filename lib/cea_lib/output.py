from importlib.resources import path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import pylab as pyl
import pywt
import numpy as np
import math
from lib.cea_lib.general import *


def imageplot(image, str='', sbpt=[], cmap='grey', colorbar=False, title_position="top", save=False, path="../outputs/img/imageplot.png", text_fontsize=None, colorbar_tick_fontsize=None):
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
        text_fontsize (int, optional): taille du texte associé à l'image.
        colorbar_tick_fontsize (int, optional): taille des graduations de la colorbar.

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
        cbar = plt.colorbar(imgplot, cax=cax)
        if colorbar_tick_fontsize is not None:
            cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

    ax.axis('off')

    if str != '':
        if title_position == "top":
            ax.set_title(str, fontsize=text_fontsize)
        elif title_position == "left":
            ax.text(
                -0.1, 0.5, str,
                transform=ax.transAxes,
                rotation=90,
                va='center',
                ha='right',
                fontsize=text_fontsize
            )
        
    if save:
        plt.savefig(path, dpi=300)


def plot_sim_2d(sim, time, titre=None, save=False, path="../outputs/img/simulation_2D.png", text_fontsize=None, title_fontsize=16, colorbar_tick_fontsize=None, shared_colorbar=False, cmap='RdYlBu_r'):
    """
    Affiche une série de simulations 2D à différents instants.

    La fonction sélectionne 9 instants répartis uniformément dans le temps
    et les affiche sous forme de grille 3x3.

    Args:
        sim (ndarray): simulations de forme (n, h, w)
        time (ndarray): vecteur des temps associés
        titre (str, optional): titre de la figure
        save (bool, optional): enregistrer la figure si True
        text_fontsize (int, optional): taille des labels temporels des sous-figures
        title_fontsize (int, optional): taille du titre global
        colorbar_tick_fontsize (int, optional): taille des graduations des colorbars
        shared_colorbar (bool, optional): affiche une colorbar commune aux 9 sous-figures

    Returns:
        None
    """
    if shared_colorbar:
        fig = plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(
            3,
            3,
            figure=fig,
            wspace=0.35,
            hspace=0.08
        )

        indices = [i * sim.shape[0] // 9 for i in range(9)]
        vmin = np.nanmin(sim[indices, :, :])
        vmax = np.nanmax(sim[indices, :, :])
        imgplot = None
        axes = []

        for i, nb in enumerate(indices):
            row, col = divmod(i, 3)
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)
            imgplot = ax.imshow(
                sim[nb, :, :],
                interpolation='nearest',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )
            ax.axis('off')
            ax.text(
                -0.1, 0.5, f"t={time[nb]:.2f}",
                transform=ax.transAxes,
                rotation=90,
                va='center',
                ha='right',
                fontsize=text_fontsize
            )

        fig.suptitle(titre, fontsize=title_fontsize)
        fig.subplots_adjust(top=0.92, right=0.88)

        right_axes = axes[2::3]
        right_edge = max(ax.get_position().x1 for ax in right_axes)
        bottom = min(ax.get_position().y0 for ax in axes)
        top = max(ax.get_position().y1 for ax in axes)
        colorbar_pad = 0.012
        colorbar_width = 0.014
        cax = fig.add_axes([right_edge + colorbar_pad, bottom, colorbar_width, top - bottom])

        cbar = fig.colorbar(imgplot, cax=cax)
        if colorbar_tick_fontsize is not None:
            cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
    else:
        plt.figure(figsize=(15, 15))

        for i in range(9):
            nb = i * sim.shape[0] // 9
            imageplot(
                sim[nb, :, :], #reverse(sim[nb, :, :])
                f"t={time[nb]:.2f}",
                [3, 3, i + 1],
                cmap=cmap,
                colorbar=True,
                title_position="left",
                text_fontsize=text_fontsize,
                colorbar_tick_fontsize=colorbar_tick_fontsize
            )

        plt.suptitle(titre, fontsize=title_fontsize)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.tight_layout()
    if save:
        plt.savefig(path, dpi=300)
    plt.show()
    
    
def plot_sim_2d_from_to(sim, time, i_start, i_end, nb, titre=None, cmap='RdYlBu_r', save=False, path="../outputs/img/simulation_2D.png", text_fontsize=None, title_fontsize=16, colorbar_tick_fontsize=None, shared_colorbar=False):
    """
    Affiche dans une image 2000x2000 le nombre nb de simulations 2D également réparties entre l'indice i_start et i_end avec un pas de i_freq.

    Args:
        sim (ndarray): simulations de forme (n, h, w)
        time (ndarray): vecteur des temps associés
        i_start (int): indice de départ
        i_end (int): indice de fin
        nb (int): nombre de simulations à afficher
        titre (str, optional): titre de la figure
        save (bool, optional): enregistrer la figure si True
        text_fontsize (int, optional): taille des labels temporels des sous-figures
        title_fontsize (int, optional): taille du titre global
        colorbar_tick_fontsize (int, optional): taille des graduations des colorbars
        shared_colorbar (bool, optional): affiche une colorbar commune aux sous-figures

    Returns:
        None
    """
    h, w = sim.shape[1], sim.shape[2]
    image_ratio = w / h

    best_rows, best_cols = 1, nb
    best_score = float("inf")
    for cols in range(1, nb + 1):
        rows = math.ceil(nb / cols)
        grid_ratio = (cols * image_ratio) / rows
        empty_slots = rows * cols - nb
        score = abs(np.log(grid_ratio)) + 0.05 * empty_slots
        if score < best_score:
            best_rows, best_cols = rows, cols
            best_score = score

    fig_ratio = (best_cols * image_ratio) / best_rows
    fig_size = 20
    if fig_ratio >= 1:
        figsize = (fig_size, fig_size / fig_ratio)
    else:
        figsize = (fig_size * fig_ratio, fig_size)

    indices = np.linspace(i_start, i_end, nb, dtype=int)
    
    if shared_colorbar:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(best_rows, best_cols, wspace=0.35, hspace=0.08)
        vmin = np.nanmin(sim[indices, :, :])
        vmax = np.nanmax(sim[indices, :, :])
        axes = []
        imgplot = None

        for i, idx in enumerate(indices):
            row, col = divmod(i, best_cols)
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)
            imgplot = ax.imshow(
                sim[idx, :, :], #reverse(sim[idx, :, :])
                interpolation='nearest',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )
            ax.axis('off')
            ax.text(
                -0.1, 0.5, f"t={time[idx]:.2f}",
                transform=ax.transAxes,
                rotation=90,
                va='center',
                ha='right',
                fontsize=text_fontsize
            )

        fig.suptitle(titre, fontsize=title_fontsize)
        fig.subplots_adjust(top=0.90, right=0.90)

        right_axes = [ax for i, ax in enumerate(axes) if i % best_cols == best_cols - 1]
        right_edge = max(ax.get_position().x1 for ax in right_axes)
        bottom = min(ax.get_position().y0 for ax in axes)
        top = max(ax.get_position().y1 for ax in axes)
        colorbar_pad = 0.012
        colorbar_width = 0.014
        cax = fig.add_axes([right_edge + colorbar_pad, bottom, colorbar_width, top - bottom])
        cbar = fig.colorbar(imgplot, cax=cax)
        if colorbar_tick_fontsize is not None:
            cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
    else:
        plt.figure(figsize=figsize)

        for i, idx in enumerate(indices):
            imageplot(
                sim[idx, :, :], #reverse(sim[idx, :, :])
                f"t={time[idx]:.2f}",
                [best_rows, best_cols, i + 1],
                cmap=cmap,
                colorbar=True,
                title_position="left",
                text_fontsize=text_fontsize,
                colorbar_tick_fontsize=colorbar_tick_fontsize
            )

        plt.suptitle(titre, fontsize=title_fontsize)
        plt.tight_layout(rect=[0, 0, 1, 0.97], w_pad=1.0, h_pad=1.0)

    if save:
        plt.savefig(path, dpi=300)
    plt.show()




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





def plot_wavelet_custom_layout_2d(x, wavelet="sym6", level=3, titre="", save=False, cmap='RdYlBu_r', path="../outputs/img/wavelet_decomposition.png"):
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
        save (bool, optional): enregistrer la figure si True

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
    ax0.imshow(x, cmap=cmap)
    ax0.set_title("Original")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(coeffs[0], cmap=cmap)
    ax1.set_title(f"LL | E={energy(coeffs[0]):.2e}")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")

    # ---------------- DETAILS ---------------- #
    for l in range(1, level + 1):

        details = coeffs[l]

        ax_lh = fig.add_subplot(gs[l, 0])
        ax_lh.imshow(details[0], cmap=cmap)
        ax_lh.set_title(f"LH L{l} | E={energy(details[0]):.2e}")
        ax_lh.axis("off")

        ax_hl = fig.add_subplot(gs[l, 1])
        ax_hl.imshow(details[1], cmap=cmap)
        ax_hl.set_title(f"HL L{l} | E={energy(details[1]):.2e}")
        ax_hl.axis("off")

        ax_hh = fig.add_subplot(gs[l, 2])
        ax_hh.imshow(details[2], cmap=cmap)
        ax_hh.set_title(f"HH L{l} | E={energy(details[2]):.2e}")
        ax_hh.axis("off")

    if save:
        plt.savefig(path, dpi=300)

    plt.tight_layout()
    plt.show()
