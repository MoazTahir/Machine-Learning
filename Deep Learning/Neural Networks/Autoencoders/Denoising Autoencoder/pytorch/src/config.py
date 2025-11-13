"""Configuration for the PyTorch denoising autoencoder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def _detect_device() -> torch.device:
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(slots=True, frozen=True)
class DenoisingConfig:
    data_dir: Path = Path("artifacts/pytorch_denoising_ae/data")
    artifact_dir: Path = Path("artifacts/pytorch_denoising_ae")
    batch_size: int = 256
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 128)
    noise_std: float = 0.3
    num_workers: int = 2
    seed: int = 2024
    device: torch.device = _detect_device()

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = DenoisingConfig()
CONFIG.ensure_dirs()
