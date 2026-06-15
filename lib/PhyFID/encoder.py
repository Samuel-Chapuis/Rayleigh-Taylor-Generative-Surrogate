from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lib.PhyFID.preprocessing import prepare_images


class PhysicalFeatureAutoencoder(nn.Module):
    """
    Autoencodeur convolutionnel utilise comme extracteur de features physiques.

    Le decodeur sert uniquement pendant l'entrainement par reconstruction.
    Ensuite, seul le bottleneck produit par ``encode`` est utilise.
    """

    def __init__(self, image_shape=(1, 28, 28), feature_dim=64):
        super().__init__()
        channels, height, width = image_shape
        self.image_shape = tuple(image_shape)
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
        )

        self.decoder_input = nn.Linear(feature_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
        )
        self.output_size = (height, width)

    def encode(self, images):
        return self.encoder(images)

    def forward(self, images):
        features = self.encode(images)
        decoded = self.decoder_input(features).reshape(len(images), 128, 4, 4)
        decoded = self.decoder(decoded)
        decoded = F.interpolate(decoded, size=self.output_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(decoded)


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_encoder(
    images,
    *,
    feature_dim=64,
    epochs=20,
    batch_size=128,
    lr=1e-3,
    device=None,
    logger=None,
):
    """
    Entraine l'encodeur sur un dataset par reconstruction auto-supervisee.

    Args:
        logger: Logger optionnel compatible avec ``diffusion_lib.Logger``.
            Si fourni, la loss de reconstruction est ecrite dans le .csv et le .log.
    """
    device = get_device(device)
    images = prepare_images(images)
    image_shape = tuple(images.shape[1:])

    model = PhysicalFeatureAutoencoder(image_shape=image_shape, feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=True)
    history = []

    if logger:
        logger.info(f"PhyFID encoder training started for {epochs} epochs")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_images = 0

        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # L'image originale est la cible: pas besoin de labels physiques.
            reconstruction = model(batch)
            loss = F.mse_loss(reconstruction, batch)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch)
            n_images += len(batch)

        mean_loss = total_loss / max(n_images, 1)
        history.append(mean_loss)
        log_string = f"Epoch {epoch + 1:03d}/{epochs} - reconstruction_loss={mean_loss:.6f}"
        print(log_string)

        if logger:
            logger.log_epoch(epoch + 1, {"reconstruction_loss": mean_loss})
            logger.info(log_string)

    if logger:
        logger.log_experiment_end("PhyFID encoder training finished")

    model.eval()
    return model, history


def save_encoder(model, encoder_path, *, history=None):
    """
    Sauvegarde les poids et la configuration minimale pour recharger l'encodeur.
    """
    encoder_path = Path(encoder_path)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_shape": model.image_shape,
            "feature_dim": model.feature_dim,
            "history": history or [],
        },
        encoder_path,
    )


def load_encoder(encoder_path, *, device=None):
    """
    Recharge un encodeur sauvegarde par ``save_encoder``.
    """
    device = get_device(device)
    checkpoint = torch.load(encoder_path, map_location=device)
    model = PhysicalFeatureAutoencoder(
        image_shape=tuple(checkpoint["image_shape"]),
        feature_dim=int(checkpoint["feature_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def encode_dataset(images, encoder, *, batch_size=128, device=None):
    """
    Encode un dataset complet en vecteurs de features.
    """
    device = get_device(device)
    images = prepare_images(images)
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    features = []

    encoder.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            features.append(encoder.encode(batch).cpu().numpy())

    return np.concatenate(features, axis=0)
