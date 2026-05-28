from tqdm.auto import tqdm
import torch
import torch.nn as nn


def training_loop(ddpm, loader, n_epochs, optim, device, display=None, store_path="ddpm_model.pt"):
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

    Returns:
        list[torch.Tensor]: Historique des pertes calculées à chaque itération.
    """
    mse = nn.MSELoss()
    best_loss = float("inf")
    loss_list = []
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            x0 = batch[0].to(device)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            loss_list.append(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        if display:
            display.show_images(ddpm.sample(device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)
    
    return loss_list