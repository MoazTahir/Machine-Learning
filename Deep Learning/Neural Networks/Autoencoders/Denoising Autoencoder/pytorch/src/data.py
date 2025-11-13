"""Dataloaders for the PyTorch denoising autoencoder."""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import DenoisingConfig
from .utils import set_seed


class NoisyFashionMNIST(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Wrap Fashion-MNIST to add Gaussian noise on the fly."""

    def __init__(self, base: datasets.FashionMNIST, noise_std: float) -> None:
        self.base = base
        self.noise_std = noise_std

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        clean, _ = self.base[index]
        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, -1.0, 1.0)
        return noisy, clean


def _transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )


def build_dataloaders(config: DenoisingConfig) -> Tuple[DataLoader, DataLoader]:
    set_seed(config.seed)
    transform = _transform()

    train_base = datasets.FashionMNIST(
        root=str(config.data_dir), train=True, download=True, transform=transform
    )
    val_base = datasets.FashionMNIST(
        root=str(config.data_dir), train=False, download=True, transform=transform
    )

    train_dataset = NoisyFashionMNIST(train_base, noise_std=config.noise_std)
    val_dataset = NoisyFashionMNIST(val_base, noise_std=config.noise_std)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
