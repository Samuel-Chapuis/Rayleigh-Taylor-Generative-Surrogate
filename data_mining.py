import os
import numpy as np
import torch

from cea_lib.data_loader import *


DATA = "data/RTCEA_bimode.hdf5"
SIZE = 28

# Output root folder for a MNIST-compatible dataset
OUTPUT_ROOT = "data/RT28"
TRAIN_SPLIT = 0.9
SEED = 0
ALLOW_OVERWRITE = True

# Label strategy for MNIST format (labels must be int64)
LABEL_MODE = "constant"  # "constant" or "column_bins"
LABEL_CONSTANT = 0
LABEL_COLUMN = 0
LABEL_BINS = 10


def to_uint8_images(data: np.ndarray) -> np.ndarray:
    images = np.empty_like(data, dtype=np.uint8)
    for i in range(data.shape[0]):
        img = data[i]
        img_min = float(np.min(img))
        img_max = float(np.max(img))
        denom = img_max - img_min
        if denom <= 0:
            images[i] = np.zeros_like(img, dtype=np.uint8)
        else:
            scaled = (img - img_min) / denom
            images[i] = (scaled * 255.0).astype(np.uint8)
    return images


def build_labels(labels: np.ndarray, n: int) -> np.ndarray:
    if LABEL_MODE == "constant":
        return np.full((n,), LABEL_CONSTANT, dtype=np.int64)

    if LABEL_MODE == "column_bins":
        column = labels[:, LABEL_COLUMN].astype(np.float64)
        edges = np.linspace(column.min(), column.max(), LABEL_BINS + 1)
        binned = np.digitize(column, edges[1:-1], right=False)
        return binned.astype(np.int64)

    raise ValueError(f"Unsupported LABEL_MODE: {LABEL_MODE}")


def save_mnist_processed(root: str,
                         train_images: np.ndarray,
                         train_labels: np.ndarray,
                         test_images: np.ndarray,
                         test_labels: np.ndarray) -> None:
    processed_dir = os.path.join(root, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    train_path = os.path.join(processed_dir, "training.pt")
    test_path = os.path.join(processed_dir, "test.pt")

    if not ALLOW_OVERWRITE and (os.path.exists(train_path) or os.path.exists(test_path)):
        raise FileExistsError(
            "Processed MNIST files already exist. "
            "Set ALLOW_OVERWRITE=True to replace them."
        )

    torch.save(
        (torch.from_numpy(train_images), torch.from_numpy(train_labels)),
        train_path
    )
    torch.save(
        (torch.from_numpy(test_images), torch.from_numpy(test_labels)),
        test_path
    )


if __name__ == "__main__":
    o_data, o_labels = load_RTCEA(DATA)
    data, labels = data_preprocessing(o_data, o_labels, resize=SIZE)
    print(data.shape, labels.shape)

    images = to_uint8_images(data)
    targets = build_labels(labels, images.shape[0])

    rng = np.random.default_rng(SEED)
    indices = np.arange(images.shape[0])
    rng.shuffle(indices)

    train_count = int(images.shape[0] * TRAIN_SPLIT)
    train_idx = indices[:train_count]
    test_idx = indices[train_count:]

    train_images = images[train_idx]
    train_labels = targets[train_idx]
    test_images = images[test_idx]
    test_labels = targets[test_idx]

    save_mnist_processed(
        OUTPUT_ROOT,
        train_images,
        train_labels,
        test_images,
        test_labels
    )

    print(
        f"Saved MNIST-compatible dataset to {OUTPUT_ROOT} "
        f"(train={train_images.shape[0]}, test={test_images.shape[0]})."
    )
