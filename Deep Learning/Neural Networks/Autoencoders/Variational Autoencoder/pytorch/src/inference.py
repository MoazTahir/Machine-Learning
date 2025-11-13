"""Inference helpers for the PyTorch variational autoencoder."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import torch

from .config import CONFIG, VAEConfig
from .model import VariationalAutoencoder


def load_model(
    config: VAEConfig = CONFIG,
    checkpoint_path: Path | None = None,
) -> VariationalAutoencoder:
    path = checkpoint_path or (config.artifact_dir / "variational_autoencoder.pt")
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Train the model first with train.py."
        )

    model = VariationalAutoencoder(latent_dim=config.latent_dim, hidden_dims=config.hidden_dims)
    device = config.device
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device)
    model.eval()
    return model


def reconstruct(
    images: Sequence[torch.Tensor],
    model: VariationalAutoencoder | None = None,
    config: VAEConfig = CONFIG,
) -> List[torch.Tensor]:
    if model is None:
        model = load_model(config=config)
    device = config.device
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        recon, _, _ = model(batch)
    return [img.detach().cpu() for img in recon]


def sample(
    num_samples: int,
    model: VariationalAutoencoder | None = None,
    config: VAEConfig = CONFIG,
) -> torch.Tensor:
    if model is None:
        model = load_model(config=config)
    with torch.no_grad():
        return model.generate(num_samples, config.device).cpu()
