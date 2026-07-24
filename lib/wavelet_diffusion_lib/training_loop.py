from pathlib import Path
from contextlib import contextmanager

from tqdm.auto import tqdm
import torch
import torch.nn as nn


class EMA:
    """Copie exponentiellement lissée des paramètres d'un modèle."""
    def __init__(self, model, decay=0.9999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1-self.decay)
            else:
                self.shadow[key].copy_(value)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    @contextmanager
    def use_weights(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.copy_to(model)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)


def _ddpm_noise_prediction_loss(ddpm, batch, mse, device):
    """
    Calcule la loss DDPM epsilon-prediction sur un batch.
    """
    x0 = batch[0].to(device)
    n = len(x0)
    eta = torch.randn_like(x0).to(device)
    u = torch.rand(n, device=device)
    t = ddpm.final_time * u.square()

    noisy_imgs = ddpm(x0, t, eta)
    eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))
    return mse(eta_theta, eta)


def evaluate_loss(ddpm, loader, device):
    """
    Evalue la loss moyenne d'un DDPM sans mettre a jour les poids.
    """
    mse = nn.MSELoss()
    was_training = ddpm.training
    ddpm.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            loss = _ddpm_noise_prediction_loss(ddpm, batch, mse, device)
            total_loss += loss.item() * len(batch[0]) / len(loader.dataset)

    if was_training:
        ddpm.train()

    return total_loss



def training_loop(ddpm, loader, n_epochs, optim, device, display=None, store_path="ddpm_model.pt", logger=None, val_loader=None, ema_decay=0.9999):
    """
    Entraîne un modèle DDPM sur un jeu de données.

    Args:
        ddpm (DDPM): Modèle de diffusion à entraîner.
        loader (torch.utils.data.DataLoader): Chargeur fournissant les lots
            d'images d'entraînement.
        n_epochs (int): Nombre d'époques d'entraînement.
        optim (torch.optim.Optimizer): Optimiseur utilisé pour la mise à jour
            des paramètres.
        device (torch.device): Dispositif de calcul utilisé pendant l'entraînement.
        display (ImageVisualizer | None, optional): Visualiseur optionnel pour
            afficher des échantillons générés en fin d'époque.
        store_path (str, optional): Chemin de sauvegarde du meilleur modèle.

    Args:
        logger (Logger | None, optional): Logger optionnel pour écrire le déroulé
            de l'expérience et les métriques d'époque.
        val_loader (torch.utils.data.DataLoader | None, optional): Chargeur de
            validation. Si fourni, le meilleur modèle est sauvegardé sur la
            loss de validation plutôt que sur la loss d'entraînement.

    Returns:
        list[float]: Historique des pertes calculées à chaque itération.
    """
    mse = nn.MSELoss()
    best_loss = float("inf")
    loss_list = []
    ema = EMA(ddpm, ema_decay)

    if logger:
        logger.info(f"Training started for {n_epochs} epochs")

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            ddpm.train()
            loss = _ddpm_noise_prediction_loss(ddpm, batch, mse, device)
            loss_value = loss.item()
            loss_list.append(loss_value)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            ema.update(ddpm)

            epoch_loss += loss_value * len(batch[0]) / len(loader.dataset)

        selection_loader = val_loader if val_loader is not None else loader
        with ema.use_weights(ddpm):
            selection_loss = evaluate_loss(ddpm, selection_loader, device)

        # Display images generated at this epoch
        if display:
            with ema.use_weights(ddpm):
                display.show_images(ddpm.sample(device=device), f"EMA images generated at epoch {epoch + 1}")

        log_string = f"Train loss at epoch {epoch + 1}: {epoch_loss:.3f}"
        if val_loader is not None:
            log_string += f" | EMA val loss: {selection_loss:.3f}"
        else:
            log_string += f" | EMA train loss: {selection_loss:.3f}"

        # Storing the model
        if best_loss > selection_loss:
            best_loss = selection_loss
            # Le checkpoint de production est l'EMA, comme dans Mallat.
            torch.save(ema.shadow, store_path)
            log_string += " --> Best model ever (stored)"

        if logger:
            metrics = {
                "train_loss": epoch_loss,
                "ema_selection_loss": selection_loss,
            }
            if val_loader is not None:
                metrics["ema_val_loss"] = selection_loss
            metrics["best_loss"] = best_loss
            metrics["stored"] = best_loss == selection_loss
            logger.log_epoch(epoch + 1, metrics)

        print(log_string)

        if logger:
            logger.info(log_string)

    if logger:
        logger.log_experiment_end("Training finished")
    
    return loss_list



def split_wave_batch(batch, device, prior_channels=1):
    """
    Separe un batch de coefficients wavelet en prior et details.

    Args:
        batch: Batch issu d'un ``TensorDataset`` contenant un tenseur
            ``(N, C, H, W)`` en premiere position.
        device: Device vers lequel deplacer les coefficients.
        prior_channels: Nombre de premiers canaux utilises comme condition.

    Returns:
        Tuple ``(prior, details)``. Pour le cas wavelet standard, ``prior`` est
        ``cA`` et ``details`` contient ``cH/cV/cD``.
    """
    coeffs = batch[0].to(device)
    prior = coeffs[:, :prior_channels]
    details = coeffs[:, prior_channels:]
    return prior, details

def wave_noise_prediction_loss(ddpm, batch, mse, device):
    """
    Calcule la loss epsilon-prediction du DDPM conditionnel wavelet.

    Le bruit est ajoute uniquement aux canaux de details. Le prior est concatene
    aux details bruites dans ``ddpm.backward`` pour conditionner la prediction.
    """
    prior, details = split_wave_batch(batch, device, prior_channels=ddpm.prior_channels)
    n = len(details)
    eta = torch.randn_like(details)
    u = torch.rand(n, device=device)
    t = ddpm.final_time * u.square()
    noisy_details = ddpm(details, t, eta)
    eta_theta = ddpm.backward(noisy_details, t.reshape(n, -1), prior)
    return mse(eta_theta, eta)

def wave_evaluate_loss(ddpm, loader, device):
    """
    Evalue la loss moyenne du DDPM conditionnel wavelet sans mise a jour.

    Le mode train/eval initial du modele est restaure apres l'evaluation.
    """
    mse = nn.MSELoss()
    was_training = ddpm.training
    ddpm.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            loss = wave_noise_prediction_loss(ddpm, batch, mse, device)
            total_loss += loss.item() * len(batch[0]) / len(loader.dataset)

    if was_training:
        ddpm.train()
    return total_loss

def wave_training_loop(ddpm, loader, n_epochs, optim, device, store_path, logger=None, val_loader=None, ema_decay=0.9999):
    """
    Entraine un DDPM conditionnel sur coefficients wavelet.

    Le modele apprend a predire le bruit sur les details conditionnellement au
    prior basse frequence. Le meilleur checkpoint est selectionne sur la loss de
    validation si ``val_loader`` est fourni, sinon sur la loss d'entrainement.

    Args:
        ddpm: Instance de ``WaveletConditionalDDPM``.
        loader: DataLoader d'entrainement.
        n_epochs: Nombre d'epoques.
        optim: Optimiseur PyTorch.
        device: Device de calcul.
        store_path: Chemin de sauvegarde du meilleur checkpoint.
        logger: Logger optionnel.
        val_loader: DataLoader de validation optionnel.

    Returns:
        Historique des pertes batch par batch.
    """
    mse = nn.MSELoss()
    best_loss = float("inf")
    loss_list = []
    ema = EMA(ddpm, ema_decay)
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info(f"Wavelet conditional training started for {n_epochs} epochs")

    for epoch in tqdm(range(n_epochs), desc="Wavelet DDPM training", colour="#00ff00"):
        epoch_loss = 0.0
        for batch in tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500"):
            ddpm.train()
            loss = wave_noise_prediction_loss(ddpm, batch, mse, device)
            loss_value = loss.item()
            loss_list.append(loss_value)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            ema.update(ddpm)

            epoch_loss += loss_value * len(batch[0]) / len(loader.dataset)

        selection_loader = val_loader if val_loader is not None else loader
        with ema.use_weights(ddpm):
            selection_loss = wave_evaluate_loss(ddpm, selection_loader, device)

        log_string = f"Train loss at epoch {epoch + 1}: {epoch_loss:.4f}"
        if val_loader is not None:
            log_string += f" | EMA val loss: {selection_loss:.4f}"
        else:
            log_string += f" | EMA train loss: {selection_loss:.4f}"

        stored = best_loss > selection_loss
        if stored:
            best_loss = selection_loss
            torch.save(ema.shadow, store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)
        if logger:
            metrics = {
                "train_loss": epoch_loss,
                "ema_selection_loss": selection_loss,
                "best_loss": best_loss,
                "stored": stored,
            }
            if val_loader is not None:
                metrics["ema_val_loss"] = selection_loss
            logger.log_epoch(epoch + 1, metrics)
            logger.info(log_string)

    if logger:
        logger.log_experiment_end("Wavelet conditional training finished")

    return loss_list
