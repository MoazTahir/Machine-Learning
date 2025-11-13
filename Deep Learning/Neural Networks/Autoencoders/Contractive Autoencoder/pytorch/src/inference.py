"""Inference utilities for the PyTorch contractive autoencoder."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import torch

from .config import CONFIG, ContractiveConfig
from .model import ContractiveAutoencoder


def load_model(
    config: ContractiveConfig = CONFIG,
    checkpoint_path: Path | None = None,
) -> ContractiveAutoencoder:
    path = checkpoint_path or (config.artifact_dir / "contractive_autoencoder.pt")
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Train the model first with train.py."
        )

    model = ContractiveAutoencoder(latent_dim=config.latent_dim, hidden_dims=config.hidden_dims)
    device = config.device
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device)
    model.eval()
    return model


def reconstruct(
    images: Sequence[torch.Tensor],
    model: ContractiveAutoencoder | None = None,
    config: ContractiveConfig = CONFIG,
) -> List[torch.Tensor]:
    if model is None:
        model = load_model(config=config)
    device = config.device
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        recon = model(batch)
    return [img.detach().cpu() for img in recon]
