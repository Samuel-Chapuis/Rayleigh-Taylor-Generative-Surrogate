import os
import torch
from lib.PhyFID import*

def load_pt_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    loaded = torch.load(path)
    return loaded


def build_phyfid_reference(
    dataset_train_path,
    dataset_val_path,
    encoder_path="out/phyfid_demo_encoder.pt",
    stats_path="out/phyfid_demo_val_stats.npz",
    train_epochs=1,
    feature_dim=32,
    batch_size=128,
    max_train_size=1024,
    max_val_size=100,
    logger=None,
):
    """
    Prepare la reference PhyFID.

    Cette fonction ne genere aucune image avec le modele de diffusion.
    Elle apprend seulement un encodeur sur TRAIN_PT, puis sauvegarde les
    statistiques de VAL_PT dans l'espace de features de cet encodeur.
    """
    phyfid_device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Chargement des datasets de reference...")
    train_images = load_pt_dataset(dataset_train_path)
    val_images = load_pt_dataset(dataset_val_path)

    print("Preparation des sous-ensembles...")
    train_subset = train_images[:min(max_train_size, len(train_images))]
    val_subset = val_images[:min(max_val_size, len(val_images))]

    print("Entrainement de l'encodeur PhyFID...")
    encoder, history = train_encoder(
        train_subset,
        feature_dim=feature_dim,
        epochs=train_epochs,
        batch_size=batch_size,
        device=phyfid_device,
        logger=logger,
    )

    print("Sauvegarde de l'encodeur...")
    save_encoder(encoder, encoder_path, history=history)

    print("Sauvegarde des statistiques de VAL_PT...")
    save_reference_statistics(
        val_subset,
        encoder_path,
        stats_path,
        batch_size=batch_size,
        device=phyfid_device,
    )

    return encoder_path, stats_path


def evaluate_phyfid_dataset(
    dataset_to_test,
    encoder_path="out/phyfid_demo_encoder.pt",
    stats_path="out/phyfid_demo_val_stats.npz",
    batch_size=128,
):
    """
    Evalue un dataset deja construit avec la reference PhyFID sauvegardee.

    Le dataset peut venir d'un DDPM, d'un fichier .pt, d'une autre methode de
    generation, etc. PhyFID ne s'occupe pas de produire ces images.
    """
    phyfid_device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Calcul du score PhyFID...")
    score = compare_to_reference(
        dataset_to_test,
        encoder_path,
        stats_path,
        batch_size=batch_size,
        device=phyfid_device,
    )
    print(f"PhyFID score: {score:.4f}")
    return score
