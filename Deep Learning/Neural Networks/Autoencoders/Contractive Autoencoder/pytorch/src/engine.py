"""Training utilities for the PyTorch contractive autoencoder."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ContractiveConfig
from .utils import psnr


def _run_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    config: ContractiveConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_psnr = 0.0
    total_contractive = 0.0
    total_batches = 0

    for batch, _ in loader:
        batch = batch.to(config.device)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)

        recon = model(batch)
        recon_loss = criterion(recon, batch)
        contractive_penalty = model.contractive_penalty(batch)  # type: ignore[attr-defined]
        loss = recon_loss + config.contractive_weight * contractive_penalty

        if optimizer:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_psnr += psnr(recon.detach(), batch)
        total_contractive += contractive_penalty.detach().item()
        total_batches += 1

    denom = max(total_batches, 1)
    return total_loss / denom, total_psnr / denom, total_contractive / denom


def train_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    config: ContractiveConfig,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float, float]:
    return _run_epoch(loader, model, criterion, config, optimizer)


def evaluate_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    config: ContractiveConfig,
) -> Tuple[float, float, float]:
    with torch.no_grad():
        return _run_epoch(loader, model, criterion, config)


def train_contractive_autoencoder(config: ContractiveConfig, model: nn.Module) -> Dict[str, List[float]]:
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
        "train_contractive": [],
        "val_loss": [],
        "val_psnr": [],
        "val_contractive": [],
    }

    best_val = float("inf")
    artifact_path = config.artifact_dir / "contractive_autoencoder.pt"

    for _ in range(config.num_epochs):
        train_loss, train_psnr, train_contractive = train_epoch(
            train_loader, model, criterion, config, optimizer
        )
        val_loss, val_psnr, val_contractive = evaluate_epoch(val_loader, model, criterion, config)

        history["train_loss"].append(train_loss)
        history["train_psnr"].append(train_psnr)
        history["train_contractive"].append(train_contractive)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_contractive"].append(val_contractive)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "config": config}, artifact_path)

    return history


from .data import build_dataloaders  # noqa: E402
