"""Inference helpers for the PyTorch denoising autoencoder."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import torch

from .config import CONFIG, DenoisingConfig
from .model import DenoisingAutoencoder


def load_model(
    config: DenoisingConfig = CONFIG,
    checkpoint_path: Path | None = None,
) -> DenoisingAutoencoder:
    path = checkpoint_path or (config.artifact_dir / "denoising_autoencoder.pt")
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Train the model first with train.py."
        )

    model = DenoisingAutoencoder(latent_dim=config.latent_dim, hidden_dims=config.hidden_dims)
    device = config.device
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device)
    model.eval()
    return model


def denoise(
    images: Sequence[torch.Tensor],
    model: DenoisingAutoencoder | None = None,
    config: DenoisingConfig = CONFIG,
) -> List[torch.Tensor]:
    if model is None:
        model = load_model(config=config)
    device = config.device
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        recon = model(batch)
    return [img.detach().cpu() for img in recon]
