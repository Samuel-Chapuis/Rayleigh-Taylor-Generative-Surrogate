import torch


def prepare_images(images):
    """
    Convertit des images en tenseur NCHW float dans [0, 1].

    Accepte les formats courants du projet: NHW uint8, NCHW, NHWC, [0, 255],
    [0, 1] ou [-1, 1].
    """
    images = torch.as_tensor(images).detach().cpu().float()

    # Les datasets .pt RT sont souvent en NHW; PyTorch Conv2d attend NCHW.
    if images.ndim == 3:
        images = images.unsqueeze(1)
    elif images.ndim == 4 and images.shape[-1] in (1, 3) and images.shape[1] not in (1, 3):
        images = images.permute(0, 3, 1, 2)

    if images.ndim != 4:
        raise ValueError(f"Format d'images non supporte: {tuple(images.shape)}")

    # Harmonise les conventions de normalisation avant l'encodage.
    min_value = float(images.min())
    max_value = float(images.max())
    if max_value > 1.5:
        images = images / 255.0
    elif min_value < -0.1:
        images = (images + 1.0) / 2.0

    return images.clamp(0.0, 1.0)
