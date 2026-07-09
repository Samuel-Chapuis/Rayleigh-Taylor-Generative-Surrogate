import os, numpy as np, torch
from lib.cea_lib.data_loader import *
from lib.cea_lib.data_augmentation import *
from lib.cea_lib.dataset_analysis import plot_dataset_overview


FILES = ["data/RTCEA_bimode.hdf5",
         "data/RTCEA_monomode_1.hdf5",
         "data/RTCEA_monomode.hdf5"]
SIZE = 64
DO_PERMUTATION = True

# ---------------------------------------- #

def data_wavelet_transform(data, wavelet='db1', level=1):
    """
    Applique la transformation en ondelettes à chaque image du dataset pour creer un dataset sous forme de coefficients d'ondelettes.

    Args:
        data (numpy.ndarray): Dataset d'images de forme (N, H, W).
        wavelet (str): Type d'ondelette à utiliser.
        level (int): Niveau de décomposition.

    Returns:
        numpy.ndarray: Dataset transformé en ondelettes de forme (N, J, C, H, W), avec J le nombre de niveaux de décomposition et C le nombre de canaux.
    """
    transformed_data = []
    for img in data:
        coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
        cA = coeffs[0]  # Approximation coefficients
        cH = coeffs[1][0]  # Horizontal detail coefficients
        cV = coeffs[1][1]  # Vertical detail coefficients
        cD = coeffs[1][2]  # Diagonal detail coefficients
        transformed_img = np.stack([cA, cH, cV, cD], axis=0)  # Stack coefficients along a new axis
        transformed_data.append(transformed_img)
    return np.array(transformed_data)

# ---------------------------------------- #

x, y = zip(*(load_RTCEA(f) for f in FILES))
data, labels = np.concatenate(x), np.concatenate(y)

plot_dataset_overview(data, labels, title="Vue d'ensemble du dataset RT-CEA", bins=50, save=True, save_path="outputs/img/dataset.png")


data, labels = data_preprocessing(data, labels, resize=SIZE)
data, labels = data_augmentater(data, labels, log=True)

wave_j1_data = data_wavelet_transform(data, wavelet='db1', level=1)
wave_j2_data = data_wavelet_transform(data, wavelet='db1', level=2)
wave_j3_data = data_wavelet_transform(data, wavelet='db1', level=3)

plot_dataset_overview(data, labels, title="Dataset RT-CEA augmanté et préparé", bins=50, save=True, save_path="outputs/img/dataset_augmente.png")

data = ((data - data.min((1,2), keepdims=True)) /
     np.maximum(np.ptp(data, axis=(1,2), keepdims=True), 1e-8) * 255).astype(np.uint8)

if DO_PERMUTATION:
     p = np.random.permutation(len(data))
     n_train = int(0.8 * len(data))
     n_val   = int(0.1 * len(data))
     n_test  = int(0.1 * len(data))

     j1_train, j1_val, j1_test = wave_j1_data[p[:n_train]], wave_j1_data[p[n_train:n_train+n_val]], wave_j1_data[p[n_train+n_val:n_train+n_val+n_test]]
     j2_train, j2_val, j2_test = wave_j2_data[p[:n_train]], wave_j2_data[p[n_train:n_train+n_val]], wave_j2_data[p[n_train+n_val:n_train+n_val+n_test]]
     j3_train, j3_val, j3_test = wave_j3_data[p[:n_train]], wave_j3_data[p[n_train:n_train+n_val]], wave_j3_data[p[n_train+n_val:n_train+n_val+n_test]]

     # train = data[p[:n_train]]
     # val   = data[p[n_train:n_train+n_val]]
     # test  = data[p[n_train+n_val:n_train+n_val+n_test]]

     plot_dataset_overview(j1_test, labels, title="Dataset RT-CEA test", bins=50, save=True, save_path="outputs/img/dataset_test.png")

     os.makedirs(f"data/RT{SIZE}/processed", exist_ok=True)
     torch.save(torch.from_numpy(j1_train), f"data/RT{SIZE}/processed/j1_training.pt")
     torch.save(torch.from_numpy(j1_val),   f"data/RT{SIZE}/processed/j1_validation.pt")
     torch.save(torch.from_numpy(j1_test),  f"data/RT{SIZE}/processed/j1_test.pt")
     torch.save(torch.from_numpy(j2_train), f"data/RT{SIZE}/processed/j2_training.pt")     
     torch.save(torch.from_numpy(j2_val),   f"data/RT{SIZE}/processed/j2_validation.pt")
     torch.save(torch.from_numpy(j2_test),  f"data/RT{SIZE}/processed/j2_test.pt")
     torch.save(torch.from_numpy(j3_train), f"data/RT{SIZE}/processed/j3_training.pt")     
     torch.save(torch.from_numpy(j3_val),   f"data/RT{SIZE}/processed/j3_validation.pt")
     torch.save(torch.from_numpy(j3_test),  f"data/RT{SIZE}/processed/j3_test.pt")


else:
     print("Export fait dans :", os.path.abspath(f"data/RT{SIZE}/processed/dataset.pt"))
     os.makedirs(f"data/RT{SIZE}/processed", exist_ok=True)
     torch.save(torch.from_numpy(wave_j1_data), f"data/RT{SIZE}/processed/o_j1dataset.pt")
     torch.save(torch.from_numpy(wave_j2_data), f"data/RT{SIZE}/processed/o_j2dataset.pt")
     torch.save(torch.from_numpy(wave_j3_data), f"data/RT{SIZE}/processed/o_j3dataset.pt")