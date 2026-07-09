import os, numpy as np, torch
from lib.cea_lib.data_loader import *
from lib.cea_lib.data_augmentation import *
from lib.cea_lib.dataset_analysis import plot_dataset_overview


FILES = ["data/RTCEA_bimode.hdf5",
         "data/RTCEA_monomode_1.hdf5",
         "data/RTCEA_monomode.hdf5"]
SIZE = 64
DO_PERMUTATION = True


x, y = zip(*(load_RTCEA(f) for f in FILES))
data, labels = np.concatenate(x), np.concatenate(y)

plot_dataset_overview(data, labels, title="Vue d'ensemble du dataset RT-CEA", bins=50, save=True, save_path="outputs/img/dataset.png")


data, labels = data_preprocessing(data, labels, resize=SIZE)
data, labels = data_augmentater(data, labels, log=True)

plot_dataset_overview(data, labels, title="Dataset RT-CEA augmanté et préparé", bins=50, save=True, save_path="outputs/img/dataset_augmente.png")

data = ((data - data.min((1,2), keepdims=True)) /
     np.maximum(np.ptp(data, axis=(1,2), keepdims=True), 1e-8) * 255).astype(np.uint8)

if DO_PERMUTATION:
     p = np.random.permutation(len(data))
     n_train = int(0.8 * len(data))
     n_val   = int(0.1 * len(data))
     n_test  = int(0.1 * len(data))

     train = data[p[:n_train]]
     val   = data[p[n_train:n_train+n_val]]
     test  = data[p[n_train+n_val:n_train+n_val+n_test]]

     plot_dataset_overview(test, labels, title="Dataset RT-CEA test", bins=50, save=True, save_path="outputs/img/dataset_test.png")

     os.makedirs(f"data/RT{SIZE}/processed", exist_ok=True)
     torch.save(torch.from_numpy(train), f"data/RT{SIZE}/processed/training.pt")
     torch.save(torch.from_numpy(val),   f"data/RT{SIZE}/processed/validation.pt")
     torch.save(torch.from_numpy(test),  f"data/RT{SIZE}/processed/test.pt")

else:
     print("Export fait dans :", os.path.abspath(f"data/RT{SIZE}/processed/dataset.pt"))
     os.makedirs(f"data/RT{SIZE}/processed", exist_ok=True)
     torch.save(torch.from_numpy(data), f"data/RT{SIZE}/processed/dataset.pt")