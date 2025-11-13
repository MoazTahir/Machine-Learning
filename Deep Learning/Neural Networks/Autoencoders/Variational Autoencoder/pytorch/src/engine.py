"""Training utilities for the PyTorch variational autoencoder."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import VAEConfig
from .utils import kl_divergence, psnr


def _run_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    config: VAEConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_psnr = 0.0
    total_kl = 0.0
    total_batches = 0

    for batch, _ in loader:
        batch = batch.to(config.device)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)

        recon, mu, logvar = model(batch)
        recon_loss = criterion(recon, batch)
        kl_term = kl_divergence(mu, logvar)
        loss = recon_loss + config.kl_weight * kl_term

        if optimizer:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_psnr += psnr(recon.detach(), batch)
        total_kl += kl_term.detach().item()
        total_batches += 1

    denom = max(total_batches, 1)
    return total_loss / denom, total_psnr / denom, total_kl / denom


def train_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    config: VAEConfig,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float, float]:
    return _run_epoch(loader, model, criterion, config, optimizer)


def evaluate_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    config: VAEConfig,
) -> Tuple[float, float, float]:
    with torch.no_grad():
        return _run_epoch(loader, model, criterion, config)


def train_variational_autoencoder(config: VAEConfig, model: nn.Module) -> Dict[str, List[float]]:
    train_loader, val_loader = build_dataloaders(config)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_psnr": [],
        "train_kl": [],
        "val_loss": [],
        "val_psnr": [],
        "val_kl": [],
    }

    best_val = float("inf")
    artifact_path = config.artifact_dir / "variational_autoencoder.pt"

    for _ in range(config.num_epochs):
        train_loss, train_psnr, train_kl = train_epoch(train_loader, model, criterion, config, optimizer)
        val_loss, val_psnr, val_kl = evaluate_epoch(val_loader, model, criterion, config)

        history["train_loss"].append(train_loss)
        history["train_psnr"].append(train_psnr)
        history["train_kl"].append(train_kl)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_kl"].append(val_kl)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "config": config}, artifact_path)

    return history


from .data import build_dataloaders  # noqa: E402
