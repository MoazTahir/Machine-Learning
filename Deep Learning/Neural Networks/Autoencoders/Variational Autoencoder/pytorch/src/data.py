"""Dataset utilities for the PyTorch variational autoencoder."""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import VAEConfig
from .utils import set_seed


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )


def build_dataloaders(config: VAEConfig) -> Tuple[DataLoader, DataLoader]:
    set_seed(config.seed)
    transform = _build_transform()

    train_ds = datasets.FashionMNIST(
        root=str(config.data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    val_ds = datasets.FashionMNIST(
        root=str(config.data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
