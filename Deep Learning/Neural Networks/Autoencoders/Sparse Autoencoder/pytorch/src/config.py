"""Configuration for the PyTorch sparse autoencoder."""
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
class SparseConfig:
    data_dir: Path = Path("artifacts/pytorch_sparse_ae/data")
    artifact_dir: Path = Path("artifacts/pytorch_sparse_ae")
    batch_size: int = 256
    num_epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 128)
    sparsity_target: float = 0.05
    sparsity_weight: float = 1e-3
    num_workers: int = 2
    seed: int = 1339
    device: torch.device = _detect_device()

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = SparseConfig()
CONFIG.ensure_dirs()
