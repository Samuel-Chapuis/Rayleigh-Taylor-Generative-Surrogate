import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def plot_dataset_overview(
    data,
    labels=None,
    label_names=None,
    title="Vue d'ensemble du dataset",
    bins=50,
    sample_count=6,
    cmap="RdYlBu_r",
    save=False,
    save_path=None,
):
    """
    Affiche un tableau de bord synthétique pour comprendre rapidement un dataset.

    Args:
        data (ndarray): données de forme (N, X, Y), (N, C, X, Y) ou équivalent.
        labels (ndarray, optional): labels associés aux données.
        label_names (list, optional): noms des colonnes de labels.
        title (str, optional): titre de la figure.
        bins (int, optional): nombre de bins pour les histogrammes.
        sample_count (int, optional): nombre d'exemples à afficher.
        cmap (str, optional): colormap pour les images.
        save (bool, optional): enregistre la figure si True.
        save_path (str, optional): chemin de sauvegarde. Par défaut basé sur le titre.

    Returns:
        None
    """
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError("Le dataset est vide.")

    finite_data = data[np.isfinite(data)]
    if finite_data.size == 0:
        raise ValueError("Le dataset ne contient aucune valeur finie.")

    labels_array = None
    if labels is not None:
        labels_array = np.asarray(labels)
        if labels_array.ndim == 1:
            labels_array = labels_array.reshape(-1, 1)

    n_samples = data.shape[0]
    spatial_shape = data.shape[1:]
    if label_names is None and labels_array is not None:
        default_names = ["Atwood", "Temps", "L", "Translation X", "Flip"]
        label_names = [
            default_names[i] if i < len(default_names) else f"Label {i}"
            for i in range(labels_array.shape[1])
        ]
    elif labels_array is not None:
        label_names = list(label_names)
        label_names += [
            f"Label {i}"
            for i in range(len(label_names), labels_array.shape[1])
        ]

    sample_means = np.nanmean(data.reshape(n_samples, -1), axis=1)
    sample_stds = np.nanstd(data.reshape(n_samples, -1), axis=1)
    mean_image = None
    std_image = None
    if data.ndim >= 3:
        mean_image = np.nanmean(data, axis=0)
        std_image = np.nanstd(data, axis=0)
        if mean_image.ndim == 3 and mean_image.shape[0] in (1, 3, 4):
            mean_image = np.moveaxis(mean_image, 0, -1)
            std_image = np.moveaxis(std_image, 0, -1)
        elif mean_image.ndim > 2:
            mean_image = np.nanmean(mean_image, axis=0)
            std_image = np.nanmean(std_image, axis=0)

    fig = plt.figure(figsize=(28, 16), constrained_layout=True)
    fig.patch.set_facecolor("#f7f7f2")
    fig.suptitle(title, fontsize=22, fontweight="bold", color="#17202a")

    gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[0.95, 1.3, 1.3, 2.4])

    ax_info = fig.add_subplot(gs[0, :2])
    ax_info.set_facecolor("#ffffff")
    ax_info.axis("off")

    label_summary = "Aucun label fourni"
    if labels_array is not None:
        unique_by_col = [
            len(np.unique(labels_array[:, i][np.isfinite(labels_array[:, i])]))
            for i in range(labels_array.shape[1])
        ]
        label_summary = (
            f"{labels_array.shape[1]} colonne(s) | "
            f"uniques: {', '.join(str(v) for v in unique_by_col)}"
        )

    stats_lines = [
        f"Taille des datas : N={n_samples} | shape={data.shape}",
        f"Dimensions spatiales : {spatial_shape}",
        f"Nombre de labels : {label_summary}",
        f"Moyenne globale : {np.nanmean(finite_data):.4g}",
        f"Ecart type global : {np.nanstd(finite_data):.4g}",
        f"Min / Max : {np.nanmin(finite_data):.4g} / {np.nanmax(finite_data):.4g}",
        f"Mediane : {np.nanmedian(finite_data):.4g}",
        f"Q1 / Q3 : {np.nanpercentile(finite_data, 25):.4g} / {np.nanpercentile(finite_data, 75):.4g}",
        f"Valeurs NaN : {np.isnan(data).sum()}",
        f"Valeurs nulles : {np.sum(data == 0)} ({100 * np.mean(data == 0):.2f}%)",
    ]
    ax_info.text(
        0.03,
        0.95,
        "\n".join(stats_lines),
        va="top",
        ha="left",
        fontsize=11,
        color="#17202a",
        linespacing=1.45,
        transform=ax_info.transAxes,
    )
    ax_info.set_title("Résumé", loc="left", fontsize=14, fontweight="bold")

    ax_hist = fig.add_subplot(gs[0, 2:])
    ax_hist.hist(finite_data.ravel(), bins=bins, color="#2f6690", alpha=0.85)
    ax_hist.axvline(np.nanmean(finite_data), color="#d1495b", lw=2, label="Moyenne")
    ax_hist.axvline(np.nanmedian(finite_data), color="#edae49", lw=2, label="Mediane")
    ax_hist.set_title("Distribution des valeurs", loc="left", fontsize=14, fontweight="bold")
    ax_hist.set_xlabel("Valeur")
    ax_hist.set_ylabel("Frequence")
    ax_hist.legend(frameon=False)
    ax_hist.grid(alpha=0.25)

    ax_mean = fig.add_subplot(gs[1, 0])
    if mean_image is not None and mean_image.ndim == 2:
        im = ax_mean.imshow(mean_image, cmap=cmap)
        ax_mean.set_anchor("W")
        divider = make_axes_locatable(ax_mean)
        cax = divider.append_axes("right", size="4%", pad=0.03)
        fig.colorbar(im, cax=cax)
    ax_mean.set_title("Image moyenne", fontsize=13, fontweight="bold")
    ax_mean.axis("off")

    ax_std = fig.add_subplot(gs[1, 1])
    if std_image is not None and std_image.ndim == 2:
        im = ax_std.imshow(std_image, cmap="magma")
        ax_std.set_anchor("W")
        divider = make_axes_locatable(ax_std)
        cax = divider.append_axes("right", size="4%", pad=0.03)
        fig.colorbar(im, cax=cax)
    ax_std.set_title("Ecart type pixel", fontsize=13, fontweight="bold")
    ax_std.axis("off")

    ax_sample_stats = fig.add_subplot(gs[1, 2:])
    ax_sample_stats.plot(sample_means, color="#2f6690", lw=1.8, label="Moyenne par sample")
    ax_sample_stats.fill_between(
        np.arange(n_samples),
        sample_means - sample_stds,
        sample_means + sample_stds,
        color="#2f6690",
        alpha=0.18,
        label="+/- ecart type",
    )
    ax_sample_stats.set_title("Statistiques par sample", loc="left", fontsize=14, fontweight="bold")
    ax_sample_stats.set_xlabel("Index")
    ax_sample_stats.grid(alpha=0.25)
    ax_sample_stats.legend(frameon=False)

    if labels_array is not None:
        max_label_plots = min(4, labels_array.shape[1])
        for i in range(max_label_plots):
            ax = fig.add_subplot(gs[2, i])
            values = labels_array[:, i]
            values = values[np.isfinite(values)]
            unique_values = np.unique(values)
            if unique_values.size <= 20:
                counts = [np.sum(values == value) for value in unique_values]
                ax.bar(unique_values.astype(str), counts, color="#4f8a8b")
                ax.tick_params(axis="x", rotation=45)
                ax.set_ylabel("Nombre")
            else:
                ax.hist(values, bins=min(bins, 35), color="#4f8a8b", alpha=0.85)
                ax.set_ylabel("Frequence")
            ax.set_title(label_names[i], fontsize=12, fontweight="bold")
            ax.grid(alpha=0.2)
    else:
        ax = fig.add_subplot(gs[2, :])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Aucun label fourni",
            ha="center",
            va="center",
            fontsize=14,
            color="#5d6d7e",
        )

    sample_count = min(sample_count, n_samples)
    sample_indices = np.linspace(0, n_samples - 1, sample_count, dtype=int)
    sample_cols = sample_count
    sample_rows = 1
    sample_grid = gridspec.GridSpecFromSubplotSpec(
        sample_rows + 1,
        sample_cols,
        subplot_spec=gs[3, :],
        height_ratios=[0.12] + [1] * sample_rows,
        hspace=0.05,
        wspace=0.22,
    )
    ax_examples_title = fig.add_subplot(sample_grid[0, :])
    ax_examples_title.axis("off")
    ax_examples_title.text(
        0,
        0.5,
        "Exemples du dataset - images reparties regulierement entre le debut et la fin",
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#17202a",
        transform=ax_examples_title.transAxes,
    )
    for pos, idx in enumerate(sample_indices):
        row = 1 + pos // sample_cols
        col = pos % sample_cols
        ax = fig.add_subplot(sample_grid[row, col])
        sample = data[idx]
        if sample.ndim == 3 and sample.shape[0] in (1, 3, 4):
            sample = np.moveaxis(sample, 0, -1)
        elif sample.ndim > 2:
            sample = np.nanmean(sample, axis=0)
        if sample.ndim == 2:
            sample_finite = sample[np.isfinite(sample)]
            if sample_finite.size > 0:
                sample_vmin = np.nanpercentile(sample_finite, 1)
                sample_vmax = np.nanpercentile(sample_finite, 99)
            else:
                sample_vmin, sample_vmax = None, None
            im = ax.imshow(sample, cmap=cmap, vmin=sample_vmin, vmax=sample_vmax)
            ax.set_anchor("W")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.02)
            fig.colorbar(im, cax=cax)
        ax.set_title(f"Exemple #{idx}", fontsize=10)
        ax.axis("off")

    if save:
        if save_path is None:
            save_path = f"img/{title.replace(' ', '_')}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()