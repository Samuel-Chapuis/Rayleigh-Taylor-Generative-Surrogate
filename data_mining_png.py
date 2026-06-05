import os, numpy as np, torch
from cea_lib.data_loader import *

files = ["data/RTCEA_bimode.hdf5",
         "data/RTCEA_monomode_1.hdf5",
         "data/RTCEA_monomode.hdf5"]

x, y = zip(*(load_RTCEA(f) for f in files))
x, y = np.concatenate(x), np.concatenate(y)
x, _ = data_preprocessing(x, y, resize=28)

x = ((x - x.min((1,2), keepdims=True)) /
     np.maximum(np.ptp(x, axis=(1,2), keepdims=True), 1e-8) * 255).astype(np.uint8)

p = np.random.permutation(len(x))
n_train = int(0.8 * len(x))
n_val   = int(0.1 * len(x))

train = x[p[:n_train]]
val   = x[p[n_train:n_train+n_val]]
test  = x[p[n_train+n_val:]]

os.makedirs("data/RT28/processed", exist_ok=True)
torch.save(torch.from_numpy(train), "data/RT28/processed/training.pt")
torch.save(torch.from_numpy(val),   "data/RT28/processed/validation.pt")
torch.save(torch.from_numpy(test),  "data/RT28/processed/test.pt")