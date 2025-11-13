"""Model definition for a fully-connected vanilla autoencoder."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import torch
from torch import nn


class VanillaAutoencoder(nn.Module):
    """Simple fully-connected autoencoder for 28x28 grayscale images."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        input_dim = 28 * 28

        encoder_layers: list[nn.Module] = [nn.Flatten()]
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Tanh())
        decoder_layers.append(_Reshape((1, 28, 28)))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class _Reshape(nn.Module):
    """Layer that reshapes tensors inside nn.Sequential."""

    def __init__(self, shape: Tuple[int, int, int]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x.view(x.size(0), *self.shape)
