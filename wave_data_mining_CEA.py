import os, numpy as np, torch
import pywt
from lib.cea_lib.data_loader import *
from lib.cea_lib.data_augmentation import *
from lib.cea_lib.dataset_analysis import plot_dataset_overview


FILES = ["data/RTCEA_bimode.hdf5",
         "data/RTCEA_monomode_1.hdf5",
         "data/RTCEA_monomode.hdf5"]
SIZE = 64
DO_PERMUTATION = True
SPLIT_SEED = 0
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SIMULATION_LABEL_COLUMNS = (0, 6, 7)  # Atwood, phase, number of modes.

# ---------------------------------------- #

def data_wavelet_transform(data, wavelet='db1', level=1, mode='periodization'):
    """
    Applique la transformation en ondelettes à chaque image du dataset pour creer un dataset sous forme de coefficients d'ondelettes.

    Args:
        data (numpy.ndarray): Dataset d'images de forme (N, H, W).
        wavelet (str): Type d'ondelette à utiliser.
        level (int): Niveau de décomposition.
        mode (str): Convention de bord PyWavelets, identique à la reconstruction.

    Returns:
        numpy.ndarray: Dataset transformé en ondelettes de forme (N, J, C, H, W), avec J le nombre de niveaux de décomposition et C le nombre de canaux.
    """
    transformed_data = []
    for img in data:
        coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level, mode=mode)
        cA = coeffs[0]  # Approximation coefficients
        cH = coeffs[1][0]  # Horizontal detail coefficients
        cV = coeffs[1][1]  # Vertical detail coefficients
        cD = coeffs[1][2]  # Diagonal detail coefficients
        transformed_img = np.stack([cA, cH, cV, cD], axis=0)  # Stack coefficients along a new axis
        transformed_data.append(transformed_img)
    return np.array(transformed_data)

# ---------------------------------------- #

def split_indices_by_simulation(labels, ratios=SPLIT_RATIOS, rng=None):
     """
     Splitte les indices sans séparer les snapshots d'une même simulation.

     D'après data/starting_doc, les colonnes 0, 6 et 7 identifient les
     paramètres fixes d'une trajectoire: Atwood, phase et nombre de modes.
     Les colonnes temporelles et de zone de mélange varient au cours d'une
     même simulation et ne doivent donc pas servir au split.
     """
     if rng is None:
          rng = np.random.default_rng(SPLIT_SEED)

     simulation_keys = np.round(labels[:, SIMULATION_LABEL_COLUMNS].astype(float), decimals=8)
     unique_keys = np.unique(simulation_keys, axis=0)
     rng.shuffle(unique_keys)

     n_simulations = len(unique_keys)
     n_train = int(ratios[0] * n_simulations)
     n_val = int(ratios[1] * n_simulations)

     split_keys = {
          "training": unique_keys[:n_train],
          "validation": unique_keys[n_train:n_train + n_val],
          "test": unique_keys[n_train + n_val:],
     }
     split_indices = {}
     for split, keys in split_keys.items():
          if len(keys) == 0:
               split_indices[split] = np.array([], dtype=np.int64)
               continue
          mask = (simulation_keys[:, None, :] == keys[None, :, :]).all(axis=2).any(axis=1)
          split_indices[split] = np.flatnonzero(mask)
     return split_indices


def load_split_by_simulation(files):
     rng = np.random.default_rng(SPLIT_SEED)
     split_data = {"training": [], "validation": [], "test": []}
     split_labels = {"training": [], "validation": [], "test": []}

     for file in files:
          data, labels = load_RTCEA(file)
          indices_by_split = split_indices_by_simulation(labels, rng=rng)
          for split, indices in indices_by_split.items():
               split_data[split].append(data[indices])
               split_labels[split].append(labels[indices])

     merged_data = {
          split: np.concatenate(chunks, axis=0)
          for split, chunks in split_data.items()
     }
     merged_labels = {
          split: np.concatenate(chunks, axis=0)
          for split, chunks in split_labels.items()
     }
     return merged_data, merged_labels


def preprocess_and_augment(data, labels, log=False):
     data, labels = data_preprocessing(data, labels, resize=SIZE)
     data, labels = data_augmentater(data, labels, log=log)
     return data, labels


def to_uint8(data):
     return ((data - data.min((1, 2), keepdims=True)) /
          np.maximum(np.ptp(data, axis=(1, 2), keepdims=True), 1e-8) * 255).astype(np.uint8)


split_data, split_labels = load_split_by_simulation(FILES)
data = np.concatenate([split_data["training"], split_data["validation"], split_data["test"]], axis=0)
labels = np.concatenate([split_labels["training"], split_labels["validation"], split_labels["test"]], axis=0)

plot_dataset_overview(data, labels, title="Vue d'ensemble du dataset RT-CEA", bins=50, save=True, save_path="outputs/img/dataset.png")


train, train_labels = preprocess_and_augment(split_data["training"], split_labels["training"], log=True)
val, val_labels = preprocess_and_augment(split_data["validation"], split_labels["validation"])
test, test_labels = preprocess_and_augment(split_data["test"], split_labels["test"])

wave_j1_train = data_wavelet_transform(train, wavelet='db1', level=1)
wave_j1_val = data_wavelet_transform(val, wavelet='db1', level=1)
wave_j1_test = data_wavelet_transform(test, wavelet='db1', level=1)
wave_j2_train = data_wavelet_transform(train, wavelet='db1', level=2)
wave_j2_val = data_wavelet_transform(val, wavelet='db1', level=2)
wave_j2_test = data_wavelet_transform(test, wavelet='db1', level=2)
wave_j3_train = data_wavelet_transform(train, wavelet='db1', level=3)
wave_j3_val = data_wavelet_transform(val, wavelet='db1', level=3)
wave_j3_test = data_wavelet_transform(test, wavelet='db1', level=3)

data = np.concatenate([train, val, test], axis=0)
labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)

plot_dataset_overview(data, labels, title="Dataset RT-CEA augmanté et préparé", bins=50, save=True, save_path="outputs/img/dataset_augmente.png")

data = to_uint8(data)

if DO_PERMUTATION:
     plot_dataset_overview(test, test_labels, title="Dataset RT-CEA test", bins=50, save=True, save_path="outputs/img/dataset_test.png")

     os.makedirs(f"data/RT{SIZE}/processed", exist_ok=True)
     torch.save(torch.from_numpy(wave_j1_train), f"data/RT{SIZE}/processed/j1_training.pt")
     torch.save(torch.from_numpy(wave_j1_val),   f"data/RT{SIZE}/processed/j1_validation.pt")
     torch.save(torch.from_numpy(wave_j1_test),  f"data/RT{SIZE}/processed/j1_test.pt")
     torch.save(torch.from_numpy(wave_j2_train), f"data/RT{SIZE}/processed/j2_training.pt")     
     torch.save(torch.from_numpy(wave_j2_val),   f"data/RT{SIZE}/processed/j2_validation.pt")
     torch.save(torch.from_numpy(wave_j2_test),  f"data/RT{SIZE}/processed/j2_test.pt")
     torch.save(torch.from_numpy(wave_j3_train), f"data/RT{SIZE}/processed/j3_training.pt")     
     torch.save(torch.from_numpy(wave_j3_val),   f"data/RT{SIZE}/processed/j3_validation.pt")
     torch.save(torch.from_numpy(wave_j3_test),  f"data/RT{SIZE}/processed/j3_test.pt")


else:
     print("Export fait dans :", os.path.abspath(f"data/RT{SIZE}/processed/dataset.pt"))
     os.makedirs(f"data/RT{SIZE}/processed", exist_ok=True)
     torch.save(torch.from_numpy(np.concatenate([wave_j1_train, wave_j1_val, wave_j1_test], axis=0)), f"data/RT{SIZE}/processed/o_j1dataset.pt")
     torch.save(torch.from_numpy(np.concatenate([wave_j2_train, wave_j2_val, wave_j2_test], axis=0)), f"data/RT{SIZE}/processed/o_j2dataset.pt")
     torch.save(torch.from_numpy(np.concatenate([wave_j3_train, wave_j3_val, wave_j3_test], axis=0)), f"data/RT{SIZE}/processed/o_j3dataset.pt")
