"""Training and evaluation loops for the vanilla autoencoder."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import AutoencoderConfig
from .utils import psnr


def _run_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0
    psnr_accum = 0.0

    for batch, _ in loader:
        batch = batch.to(device)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        recon = model(batch)
        loss = criterion(recon, batch)
        if optimizer:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        psnr_accum += psnr(recon.detach(), batch)
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    avg_psnr = psnr_accum / max(total_batches, 1)
    return avg_loss, avg_psnr


def train_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    return _run_epoch(loader, model, criterion, device, optimizer)


def evaluate_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    with torch.no_grad():
        return _run_epoch(loader, model, criterion, device)


def train_autoencoder(config: AutoencoderConfig, model: nn.Module) -> Dict[str, List[float]]:
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
        "val_loss": [],
        "val_psnr": [],
    }

    best_val = float("inf")
    artifact_path = config.artifact_dir / "vanilla_autoencoder.pt"

    for _ in range(config.num_epochs):
        train_loss, train_psnr = train_epoch(train_loader, model, criterion, config.device, optimizer)
        val_loss, val_psnr = evaluate_epoch(val_loader, model, criterion, config.device)

        history["train_loss"].append(train_loss)
        history["train_psnr"].append(train_psnr)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "config": config}, artifact_path)

    return history


from .data import build_dataloaders  # noqa: E402
