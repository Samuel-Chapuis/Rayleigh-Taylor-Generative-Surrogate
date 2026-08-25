"""Losses and training loops specific to conditional wavelet diffusion."""

from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm


def split_wave_batch(batch, device, prior_channels=1):
    """Split [cA, details] coefficients into conditioning and diffused channels."""
    coeffs = batch[0].to(device)
    return coeffs[:, :prior_channels], coeffs[:, prior_channels:]


def wave_noise_prediction_loss(ddpm, batch, mse, device):
    prior, details = split_wave_batch(batch, device, ddpm.prior_channels)
    eta = torch.randn_like(details)
    t = torch.randint(0, ddpm.n_steps, (len(details),), device=device)
    return mse(ddpm.backward(ddpm(details, t, eta), t.reshape(len(details), -1), prior), eta)


def wave_evaluate_loss(ddpm, loader, device):
    mse = nn.MSELoss()
    was_training = ddpm.training
    ddpm.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            total_loss += wave_noise_prediction_loss(ddpm, batch, mse, device).item() * len(batch[0])
    if was_training:
        ddpm.train()
    return total_loss / max(len(loader.dataset), 1)


def wave_training_loop(ddpm, loader, n_epochs, optim, device, store_path, logger=None, val_loader=None):
    """Train a conditional DDPM on wavelet detail coefficients."""
    mse = nn.MSELoss()
    best_loss = float("inf")
    losses = []
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info(f"Wavelet conditional DDPM training started for {n_epochs} epochs")
    for epoch in tqdm(range(n_epochs), desc="Wavelet DDPM training", colour="#00ff00"):
        ddpm.train()
        total_loss = 0.0
        for batch in tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500"):
            loss = wave_noise_prediction_loss(ddpm, batch, mse, device)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            loss_value = loss.item()
            losses.append(loss_value)
            total_loss += loss_value * len(batch[0])
        train_loss = total_loss / max(len(loader.dataset), 1)
        val_loss = wave_evaluate_loss(ddpm, val_loader, device) if val_loader is not None else None
        selection_loss = train_loss if val_loss is None else val_loss
        stored = selection_loss < best_loss
        if stored:
            best_loss = selection_loss
            torch.save(ddpm.state_dict(), store_path)
        if logger:
            metrics = {"train_loss": train_loss, "best_loss": best_loss, "stored": stored}
            if val_loss is not None:
                metrics["val_loss"] = val_loss
            logger.log_epoch(epoch + 1, metrics)
            logger.info(f"Wavelet DDPM epoch {epoch + 1}: train={train_loss:.6f}")
        print(f"Wavelet DDPM epoch {epoch + 1}: train={train_loss:.6f}" +
              (f" | val={val_loss:.6f}" if val_loss is not None else "") +
              (" --> Best model stored" if stored else ""))
    if logger:
        logger.log_experiment_end("Wavelet conditional DDPM training finished")
    return losses


def wave_sgm_loss(sgm, batch, mse, device, generator=None):
    """Denoising score-matching loss on wavelet details conditioned by cA."""
    prior, details = split_wave_batch(batch, device, sgm.prior_channels)
    n = details.shape[0]
    t = torch.rand(n, device=device, dtype=details.dtype, generator=generator)
    t = sgm.eps_time + (1.0 - sgm.eps_time) * t
    eps = torch.randn(details.shape, device=device, dtype=details.dtype, generator=generator)
    noisy_details = sgm(details, t, eps)
    prediction = sgm.network(torch.cat((prior, noisy_details), dim=1), t)
    return mse(prediction, sgm.prediction_target(details, t, eps))


def wave_sgm_evaluate_loss(sgm, loader, device, seed=12345):
    mse = nn.MSELoss()
    was_training = sgm.training
    sgm.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    total_loss = 0.0
    n_total = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = len(batch[0])
            total_loss += wave_sgm_loss(sgm, batch, mse, device, generator).item() * batch_size
            n_total += batch_size
    if was_training:
        sgm.train()
    return total_loss / max(n_total, 1)


def wave_sgm_training_loop(
    sgm, loader, n_epochs, optim, device, store_path, logger=None, val_loader=None, grad_clip=None,
):
    """Train a conditional VP-SGM on wavelet details."""
    mse = nn.MSELoss()
    start_epoch = logger.last_epoch() if logger else 0
    best_loss = logger.min_numeric_column("best_loss") if logger else float("inf")
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info(f"Wavelet SGM training started for {n_epochs} epochs")
    end_epoch = start_epoch + n_epochs
    for epoch in tqdm(range(start_epoch, end_epoch), desc="Wavelet SGM training", colour="#00ff00"):
        sgm.train()
        total_loss = 0.0
        n_total = 0
        for batch in tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{end_epoch}", colour="#005500"):
            loss = wave_sgm_loss(sgm, batch, mse, device)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(sgm.parameters(), grad_clip)
            optim.step()
            batch_size = len(batch[0])
            total_loss += loss.item() * batch_size
            n_total += batch_size
        train_loss = total_loss / max(n_total, 1)
        val_loss = wave_sgm_evaluate_loss(sgm, val_loader, device) if val_loader is not None else None
        selection_loss = train_loss if val_loss is None else val_loss
        stored = selection_loss < best_loss
        if stored:
            best_loss = selection_loss
            torch.save(sgm.state_dict(), store_path)
        if logger:
            metrics = {"train_loss": train_loss, "best_loss": best_loss, "stored": stored}
            if val_loss is not None:
                metrics["val_loss"] = val_loss
            logger.log_epoch(epoch + 1, metrics)
            logger.info(f"Wavelet SGM epoch {epoch + 1}: train={train_loss:.6f}")
        print(f"Wavelet SGM epoch {epoch + 1}: train={train_loss:.6f}" +
              (f" | val={val_loss:.6f}" if val_loss is not None else "") +
              (" --> Best model stored" if stored else ""))
    if logger:
        logger.log_experiment_end("Wavelet SGM training finished")
