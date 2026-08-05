import os, numpy as np, torch
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

data = np.concatenate([train, val, test], axis=0)
labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)

plot_dataset_overview(data, labels, title="Dataset RT-CEA augmanté et préparé", bins=50, save=True, save_path="outputs/img/dataset_augmente.png")

train = to_uint8(train)
val = to_uint8(val)
test = to_uint8(test)

if DO_PERMUTATION:
     plot_dataset_overview(test, test_labels, title="Dataset RT-CEA test", bins=50, save=True, save_path="outputs/img/dataset_test.png")

     os.makedirs(f"data/RT{SIZE}/processed", exist_ok=True)
     torch.save(torch.from_numpy(train), f"data/RT{SIZE}/processed/training.pt")
     torch.save(torch.from_numpy(val),   f"data/RT{SIZE}/processed/validation.pt")
     torch.save(torch.from_numpy(test),  f"data/RT{SIZE}/processed/test.pt")

else:
     print("Export fait dans :", os.path.abspath(f"data/RT{SIZE}/processed/dataset.pt"))
     os.makedirs(f"data/RT{SIZE}/processed", exist_ok=True)
     torch.save(torch.from_numpy(to_uint8(data)), f"data/RT{SIZE}/processed/dataset.pt")
