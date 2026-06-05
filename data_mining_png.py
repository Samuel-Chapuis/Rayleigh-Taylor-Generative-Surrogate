import os, glob, numpy as np, torch
from PIL import Image

IMG_DIR = "data/IMG28/img32"
OUT_DIR = "data/IMG28/processed"

# Chargement des images
files = sum([glob.glob(f"{IMG_DIR}/*{ext}")
             for ext in (".png", ".jpg", ".jpeg")], [])

x = np.stack([
    ((img := np.array(Image.open(f).convert("L").resize((28,28)),
                      dtype=np.float32))
      - img.min()) / max(img.max() - img.min(), 1e-8) * 255
    for f in files
]).astype(np.uint8)

# Split 80/10/10
p = np.random.permutation(len(x))
n_train = int(0.8 * len(x))
n_val = int(0.1 * len(x))

train = x[p[:n_train]]
val   = x[p[n_train:n_train+n_val]]
test  = x[p[n_train+n_val:]]

# Sauvegarde
os.makedirs(OUT_DIR, exist_ok=True)
torch.save(torch.from_numpy(train), f"{OUT_DIR}/training.pt")
torch.save(torch.from_numpy(val),   f"{OUT_DIR}/validation.pt")
torch.save(torch.from_numpy(test),  f"{OUT_DIR}/test.pt")

print(f"train={len(train)} val={len(val)} test={len(test)}")