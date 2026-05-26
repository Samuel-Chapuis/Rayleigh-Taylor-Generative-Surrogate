import os

from cea_lib.data_loader import*
from cea_lib.data_exporter import*


DATA = "data/RTCEA_bimode.hdf5"
SIZE = 32
EXPORT_FOLDER = "datasets/RT"+str(SIZE)+"/"
PREFIX = "sim"

if __name__ == "__main__":
    o_data, o_labels = load_RTCEA(DATA)
    data, labels = data_preprocessing(o_data, o_labels, resize=SIZE)
    print(data.shape, labels.shape)
    # os.makedirs(EXPORT_FOLDER, exist_ok=True)
    # export_sim_to_jpg(data, labels, folder=EXPORT_FOLDER, prefix=PREFIX)
