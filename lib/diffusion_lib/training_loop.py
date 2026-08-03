from tqdm.auto import tqdm
import torch
import torch.nn as nn
from pathlib import Path


def _ddpm_noise_prediction_loss(ddpm, batch, mse, device):
    """
    Calcule la loss DDPM epsilon-prediction sur un batch.
    """
    x0 = batch[0].to(device)
    n = len(x0)
    eta = torch.randn_like(x0).to(device)
    t = torch.randint(0, ddpm.n_steps, (n,), device=device)

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



def training_loop(ddpm, loader, n_epochs, optim, device, display=None, store_path="ddpm_model.pt", logger=None, val_loader=None):
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
            torch.save(ddpm.state_dict(), store_path)
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


def _sgm_epsilon_loss(sgm, batch, mse, device):
    """Loss stable de denoising score matching pour un VP-SDE continu.

    En mode ``v``, la cible reste de variance bornee lorsque sigma(t) tend
    vers zero. Cela evite d'entrainer directement une quantite de score dont
    l'echelle diverge comme 1 / sigma(t).
    """
    x0 = batch[0].to(device)
    n = x0.shape[0]
    t = torch.rand(n, device=device, dtype=x0.dtype)
    t = sgm.eps_time + (1.0 - sgm.eps_time) * t
    eps = torch.randn_like(x0)
    xt = sgm(x0, t, eps)
    prediction = sgm.network(xt, t)
    target = sgm.prediction_target(x0, t, eps)
    return mse(prediction, target)


def evaluate_sgm_loss(sgm, loader, device):
    """Evalue la loss SGM moyenne sur un loader, sans mise a jour."""
    mse = nn.MSELoss()
    was_training = sgm.training
    sgm.eval()
    total_loss = 0.0
    n_total = 0
    with torch.no_grad():
        for batch in loader:
            loss = _sgm_epsilon_loss(sgm, batch, mse, device)
            batch_size = len(batch[0])
            total_loss += loss.item() * batch_size
            n_total += batch_size
    if was_training:
        sgm.train()
    return total_loss / max(n_total, 1)


def sgm_training_loop(
    sgm,
    loader,
    n_epochs,
    optim,
    device,
    display=None,
    store_path="sgm_model.pt",
    logger=None,
    val_loader=None,
    grad_clip=None,
):
    """Entraine un SGM VP-SDE par minibatches et sauvegarde le meilleur modele."""
    mse = nn.MSELoss()
    start_epoch = logger.last_epoch() if logger else 0
    checkpoint_exists = Path(store_path).exists()
    best_loss = (
        logger.min_numeric_column("best_loss")
        if logger and checkpoint_exists
        else float("inf")
    )
    loss_list = []
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info(f"SGM training started for {n_epochs} epochs")

    end_epoch = start_epoch + n_epochs
    for epoch in tqdm(range(start_epoch, end_epoch), desc="SGM training", colour="#00ff00"):
        sgm.train()
        weighted_loss = 0.0
        n_total = 0
        for batch in tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{end_epoch}", colour="#005500"):
            loss = _sgm_epsilon_loss(sgm, batch, mse, device)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(sgm.parameters(), grad_clip)
            optim.step()

            batch_size = len(batch[0])
            loss_value = loss.item()
            loss_list.append(loss_value)
            weighted_loss += loss_value * batch_size
            n_total += batch_size

        train_loss = weighted_loss / max(n_total, 1)
        val_loss = evaluate_sgm_loss(sgm, val_loader, device) if val_loader is not None else None
        selection_loss = train_loss if val_loss is None else val_loss
        stored = selection_loss < best_loss
        if stored:
            best_loss = selection_loss
            torch.save(sgm.state_dict(), store_path)

        if display:
            display.show_images(sgm.sample(device=device), f"SGM epoch {epoch + 1}")

        log_string = f"SGM train loss at epoch {epoch + 1}: {train_loss:.6f}"
        if val_loss is not None:
            log_string += f" | Val loss: {val_loss:.6f}"
        if stored:
            log_string += " --> Best model stored"
        print(log_string)
        if logger:
            metrics = {"train_loss": train_loss, "best_loss": best_loss, "stored": stored}
            if val_loss is not None:
                metrics["val_loss"] = val_loss
            logger.log_epoch(epoch + 1, metrics)
            logger.info(log_string)

    if logger:
        logger.log_experiment_end("SGM training finished")
    if not Path(store_path).exists():
        # A stale CSV can contain a lower historical loss than the current
        # run while its checkpoint is unavailable. Keep the run usable.
        torch.save(sgm.state_dict(), store_path)
        if logger:
            logger.warning("No best checkpoint was available; saved the final SGM state.")
    return loss_list
