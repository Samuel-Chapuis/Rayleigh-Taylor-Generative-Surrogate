from tqdm.auto import tqdm
import torch
import torch.nn as nn


class EMA:
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


def _ddpm_noise_prediction_loss(ddpm, batch, mse, device):
    """
    Calcule la loss DDPM epsilon-prediction sur un batch.
    """
    x0 = batch[0].to(device)
    n = len(x0)
    eta = torch.randn_like(x0).to(device)
    t = ddpm.final_time * torch.rand(n, device=device).square()

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

        val_loss = evaluate_loss(ddpm, val_loader, device) if val_loader is not None else None
        selection_loss = val_loss if val_loss is not None else epoch_loss

        # Display images generated at this epoch
        if display:
            display.show_images(ddpm.sample(device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Train loss at epoch {epoch + 1}: {epoch_loss:.3f}"
        if val_loss is not None:
            log_string += f" | Val loss: {val_loss:.3f}"

        # Storing the model
        if best_loss > selection_loss:
            best_loss = selection_loss
            torch.save(ema.shadow, store_path)
            log_string += " --> Best model ever (stored)"

        if logger:
            metrics = {
                "train_loss": epoch_loss,
            }
            if val_loss is not None:
                metrics["val_loss"] = val_loss
            metrics["best_loss"] = best_loss
            metrics["stored"] = best_loss == selection_loss
            logger.log_epoch(epoch + 1, metrics)

        print(log_string)

        if logger:
            logger.info(log_string)

    if logger:
        logger.log_experiment_end("Training finished")
    
    return loss_list
