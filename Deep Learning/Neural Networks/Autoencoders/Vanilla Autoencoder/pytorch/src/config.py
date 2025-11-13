"""Configuration for the PyTorch vanilla autoencoder module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def _detect_device() -> torch.device:
    """Return the best available device with MPS preference."""
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(slots=True, frozen=True)
class AutoencoderConfig:
    data_dir: Path = Path("artifacts/pytorch_vanilla_ae/data")
    artifact_dir: Path = Path("artifacts/pytorch_vanilla_ae")
    batch_size: int = 256
    num_epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 128)
    num_workers: int = 2
    seed: int = 1337
    device: torch.device = _detect_device()

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = AutoencoderConfig()
CONFIG.ensure_dirs()
