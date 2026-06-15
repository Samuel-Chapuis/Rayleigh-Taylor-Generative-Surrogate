import csv
import os
import time
from datetime import datetime

import numpy as np


def _get_memory_usage_mb():
    """Retourne la memoire residente du processus courant en Mo."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as statm_file:
            resident_pages = int(statm_file.readline().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 2)
    except (FileNotFoundError, IndexError, OSError, ValueError):
        try:
            import resource

            memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if os.name == "posix":
                return memory_kb / 1024
            return memory_kb / (1024 ** 2)
        except (ImportError, OSError, ValueError):
            return None


def _array_size_mb(array):
    return getattr(array, "nbytes", 0) / (1024 ** 2)


def augmentation(image, label):
    """
    Effectue une augmentation de données sur une image d'entrée en appliquant 9 translation uniformes d'axe x, et aussi un mirror horizontal puis de nouveau 9 translations l'image flip. Ce qui donne un total de 20 images a exportées (19 générées + l'image d'entrée).

    Args:
        image (numpy.ndarray): Un tableau 2D contenant les données de simulation (x, y).
        label (numpy.ndarray): Un tableau 1D contenant les étiquettes associées aux simulations.

    Returns:
        tuple:
            - generated_images (ndarray): Tableau 3D contenant les images générées à partir de l'image d'entrée.
            - generated_labels (ndarray): Tableau 2D contenant les étiquettes associées aux images générées.
    """

    generated_images = []
    generated_labels = []
    # Ajouter l'image d'origine
    generated_images.append(image)
    # On modifie le label pour ajouter une dimension de translation et de flip
    original_label = np.concatenate((label, [0, 0])) # [translation_x, flip]
    generated_labels.append(original_label)

    # Générer des images en appliquant des translations uniformes sur l'axe x
    for translation in range(-5, 5):  # Translations de -5 à 5
        if translation == 0:
            flipped_image = np.fliplr(image)  # Miroir horizontal de l'image d'origine
            generated_images.append(flipped_image)
            flipped_label = np.concatenate((label, [0, 1]))  # Ajouter le flip au label
            generated_labels.append(flipped_label)
            continue  # Ignorer la translation nulle (image d'origine)
            
        translated_image = np.roll(image, shift=translation, axis=1)  # Translation sur l'axe x
        generated_images.append(translated_image)
        translated_label = np.concatenate((label, [translation, 0]))  # Ajouter la translation au label
        generated_labels.append(translated_label)

        # Générer une image en appliquant un miroir horizontal
        flipped_image = np.fliplr(translated_image)  # Miroir horizontal
        generated_images.append(flipped_image)
        flipped_label = np.concatenate((label, [translation, 1]))  # Ajouter la translation et le flip au label
        generated_labels.append(flipped_label)

    # Convertir les listes en tableaux numpy
    generated_images = np.array(generated_images)
    generated_labels = np.array(generated_labels)

    return generated_images, generated_labels


def data_augmentater(images, labels, log=False):
    """
    Effectue une augmentation de données sur l'ensemble des images d'entrée et exporte les images générées avec leurs étiquettes associées.

    Args:
        images (numpy.ndarray): Un tableau 3D contenant les données de simulation (x, y).
        labels (numpy.ndarray): Un tableau 2D contenant les étiquettes associées aux simulations.
        log (bool): Si True, cree/remplace augmantation_log.csv et y ajoute une ligne a chaque appel de augmentation.

    Returns:
        tuple:
            - generated_images_dataset (ndarray): Tableau 3D contenant les images générées à partir de l'image d'entrées.
            - generated_labels_dataset (ndarray): Tableau 2D contenant les étiquettes associées aux images générées.
    """

    generated_images_dataset = []
    generated_labels_dataset = []

    log_file = None
    log_writer = None

    if log:
        log_file = open("augmantation_log.csv", mode="w", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "timestamp",
                "image_index",
                "image_shape",
                "label_shape",
                "generated_images",
                "duration_seconds",
                "memory_before_mb",
                "memory_after_mb",
                "memory_delta_mb",
                "input_image_mb",
                "generated_images_mb",
                "generated_labels_mb",
            ],
        )
        log_writer.writeheader()

    try:
        for image_index, (image, label) in enumerate(zip(images, labels)):
            memory_before = _get_memory_usage_mb() if log else None
            start_time = time.perf_counter()

            generated_images, generated_labels = augmentation(image, label)

            duration_seconds = time.perf_counter() - start_time
            memory_after = _get_memory_usage_mb() if log else None

            if log_writer is not None:
                memory_delta = (
                    memory_after - memory_before
                    if memory_before is not None and memory_after is not None
                    else None
                )
                log_writer.writerow(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "image_index": image_index,
                        "image_shape": tuple(image.shape),
                        "label_shape": tuple(label.shape),
                        "generated_images": len(generated_images),
                        "duration_seconds": round(duration_seconds, 6),
                        "memory_before_mb": round(memory_before, 3) if memory_before is not None else "",
                        "memory_after_mb": round(memory_after, 3) if memory_after is not None else "",
                        "memory_delta_mb": round(memory_delta, 3) if memory_delta is not None else "",
                        "input_image_mb": round(_array_size_mb(image), 6),
                        "generated_images_mb": round(_array_size_mb(generated_images), 6),
                        "generated_labels_mb": round(_array_size_mb(generated_labels), 6),
                    }
                )
                log_file.flush()

            generated_images_dataset.append(generated_images)
            generated_labels_dataset.append(generated_labels)
    finally:
        if log_file is not None:
            log_file.close()

    # Convertir les listes en tableaux numpy
    generated_images_dataset = np.concatenate(generated_images_dataset, axis=0)
    generated_labels_dataset = np.concatenate(generated_labels_dataset, axis=0)

    return generated_images_dataset, generated_labels_dataset
