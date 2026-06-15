# PhyFID

`PhyFID` compare deux ensembles d'images à partir de la distance de Fréchet
entre leurs distributions de caractéristiques.

La bibliothèque propose deux métriques :

- **PhyFID** : les caractéristiques sont produites par un autoencodeur entraîné
  sur les images du domaine étudié ;

Un score faible indique que les deux distributions sont proches. Le score doit
être comparé uniquement entre expériences utilisant le même encodeur, le même
prétraitement et un nombre d'images comparable.

## Dépendances

La bibliothèque utilise notamment :

```text
torch
numpy
scipy
pytorch-fid
```

## Utilisation rapide

### 1. Créer une référence PhyFID

Cette opération entraîne un nouvel encodeur, le sauvegarde, puis calcule les
statistiques du jeu de validation.

```python
from PhyFID.utils import build_phyfid_reference

build_phyfid_reference(
    dataset_train_path="data/processed/training.pt",
    dataset_val_path="data/processed/validation.pt",
    encoder_path="out/phyfid_encoder.pt",
    stats_path="out/phyfid_val_stats.npz",
    train_epochs=20,
    feature_dim=64,
    batch_size=128,
    max_train_size=10_000,
    max_val_size=1_000,
)
```

Attention : `build_phyfid_reference` réentraîne actuellement l'encodeur à chaque
appel et écrase les fichiers indiqués.

### 2. Évaluer un dataset généré

Cette opération recharge l'encodeur et les statistiques existants. Elle
n'entraîne aucun réseau.

```python
from PhyFID.utils import evaluate_phyfid_dataset

score = evaluate_phyfid_dataset(
    generated_dataset,
    encoder_path="out/phyfid_encoder.pt",
    stats_path="out/phyfid_val_stats.npz",
    batch_size=128,
)
```

### 3. Comparer directement deux datasets

```python
from PhyFID import compare_datasets

score = compare_datasets(
    validation_dataset,
    generated_dataset,
    "out/phyfid_encoder.pt",
    batch_size=128,
)
```

Ici, les statistiques des deux datasets sont recalculées à chaque appel.

## API par module

### `preprocessing.py`

#### `prepare_images(images)`

Convertit un dataset en tenseur PyTorch `NCHW`, de type `float`, normalisé dans
`[0, 1]`.

Formats pris en charge : `NHW`, `NCHW`, `NHWC`, valeurs dans `[0, 255]`,
`[0, 1]` ou `[-1, 1]`.

```python
from PhyFID import prepare_images

images = prepare_images(raw_images)
```

### `encoder.py`

#### `PhysicalFeatureAutoencoder(image_shape=(1, 28, 28), feature_dim=64)`

Autoencodeur convolutionnel utilisé pour apprendre les caractéristiques du
domaine. Le décodeur sert pendant l'entraînement ; la méthode `encode()` fournit
ensuite les vecteurs employés par le PhyFID.

```python
from PhyFID import PhysicalFeatureAutoencoder

model = PhysicalFeatureAutoencoder((1, 28, 28), feature_dim=64)
features = model.encode(images)
reconstruction = model(images)
```

#### `get_device(device=None)`

Retourne le périphérique demandé, ou sélectionne automatiquement CUDA lorsqu'il
est disponible. Cette fonction est un utilitaire interne du module.

#### `train_encoder(...)`

Entraîne un nouvel autoencodeur par reconstruction avec une loss MSE.

```python
from PhyFID import train_encoder

encoder, history = train_encoder(
    train_images,
    feature_dim=64,
    epochs=20,
    batch_size=128,
    lr=1e-3,
    device="cuda",
    logger=None,
)
```

La fonction retourne le modèle entraîné et la liste des losses moyennes par
époque. `logger` peut être une instance compatible avec
`diffusion_lib.Logger`.

#### `save_encoder(model, encoder_path, history=None)`

Sauvegarde les poids, la forme des images, la dimension des caractéristiques et
l'historique d'entraînement.

```python
from PhyFID import save_encoder

save_encoder(encoder, "out/phyfid_encoder.pt", history=history)
```

#### `load_encoder(encoder_path, device=None)`

Reconstruit un encodeur sauvegardé et charge ses poids.

```python
from PhyFID import load_encoder

encoder, checkpoint = load_encoder(
    "out/phyfid_encoder.pt",
    device="cuda",
)
```

#### `encode_dataset(images, encoder, batch_size=128, device=None)`

Encode toutes les images sans calculer de gradients et retourne un tableau
NumPy de forme `(nombre_images, feature_dim)`.

```python
from PhyFID import encode_dataset

features = encode_dataset(images, encoder, batch_size=128)
```

### `metrics.py`

#### `feature_statistics(features)`

Calcule la moyenne `mu` et la covariance `sigma` des caractéristiques. Cette
fonction est principalement un élément interne du calcul.

```python
from PhyFID.metrics import feature_statistics

mu, sigma = feature_statistics(features)
```

#### `frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6)`

Calcule la distance de Fréchet entre deux distributions gaussiennes. `eps` sert
à stabiliser le calcul lorsque les covariances sont presque singulières.

#### `save_reference_statistics(...)`

Charge un encodeur, encode un dataset de référence et sauvegarde sa moyenne et
sa covariance dans un fichier `.npz`.

```python
from PhyFID import save_reference_statistics

mu, sigma = save_reference_statistics(
    validation_images,
    "out/phyfid_encoder.pt",
    "out/phyfid_val_stats.npz",
    batch_size=128,
)
```

#### `load_reference_statistics(stats_path)`

Recharge les tableaux `mu` et `sigma` précédemment sauvegardés.

```python
from PhyFID import load_reference_statistics

mu, sigma = load_reference_statistics("out/phyfid_val_stats.npz")
```

#### `compare_datasets(...)`

Encode deux datasets, calcule leurs statistiques puis leur distance de Fréchet.
Les deux ensembles sont donc retraités à chaque appel.

```python
from PhyFID import compare_datasets

score = compare_datasets(
    dataset_a,
    dataset_b,
    "out/phyfid_encoder.pt",
    batch_size=128,
    device="cuda",
)
```

#### `compare_to_reference(...)`

Compare un dataset aux statistiques de référence déjà sauvegardées. Cette
fonction évite de réencoder le jeu de validation à chaque évaluation.

```python
from PhyFID import compare_to_reference

score = compare_to_reference(
    generated_dataset,
    "out/phyfid_encoder.pt",
    "out/phyfid_val_stats.npz",
    batch_size=128,
    device="cuda",
)
```

### `utils.py`

#### `load_pt_dataset(path)`

Vérifie l'existence d'un fichier puis le charge avec `torch.load`.

#### `build_phyfid_reference(...)`

Enchaîne les opérations suivantes :

1. chargement des datasets d'entraînement et de validation ;
2. limitation éventuelle à `max_train_size` et `max_val_size` ;
3. entraînement d'un nouvel encodeur ;
4. sauvegarde de l'encodeur ;
5. calcul et sauvegarde des statistiques de validation.

Cette fonction sert à **créer ou recréer** une référence, pas à charger une
référence existante.

#### `evaluate_phyfid_dataset(...)`

Interface simplifiée autour de `compare_to_reference`. Elle charge les fichiers
existants, affiche le score et le retourne.

### `inception_fid.py`

#### `prepare_images_for_inception(images)`

Normalise les images dans `[0, 1]`, les convertit en `NCHW` et transforme les
images monochromes en RGB par répétition du canal.

#### `extract_inception_features(...)`

Extrait par lots les activations d'un modèle Inception fourni. Cette fonction
est principalement utilisée en interne par `calculate_inception_fid`.

#### `calculate_inception_fid(...)`

Construit l'InceptionV3 de `pytorch-fid`, extrait les caractéristiques des deux
datasets et calcule la FID standard.

## Fonctions importées directement

Les fonctions principales sont accessibles avec :

```python
from PhyFID import *
```

Cela inclut l'autoencodeur, son entraînement et sa sauvegarde, les comparaisons
PhyFID, le prétraitement et la FID Inception. Les fonctions de haut niveau
`build_phyfid_reference` et `evaluate_phyfid_dataset` doivent actuellement être
importées depuis `PhyFID.utils`.

## Bonnes pratiques

- Utiliser le même encodeur pour toutes les comparaisons PhyFID.
- Ne pas réentraîner l'encodeur entre deux modèles que l'on souhaite comparer.
- Utiliser le même nombre d'images dans chaque expérience.
- Employer un nombre d'images supérieur à `feature_dim` pour mieux estimer la
  covariance.
- Conserver ensemble le fichier de l'encodeur `.pt` et ses statistiques `.npz`.
- Ne pas comparer directement une valeur PhyFID à une FID Inception : les deux
  espaces de caractéristiques sont différents.
- Traiter la FID Inception comme une heuristique pour les images RT.
