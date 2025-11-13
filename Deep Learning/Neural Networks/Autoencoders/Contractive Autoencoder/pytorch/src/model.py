"""Model for the PyTorch contractive autoencoder."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ContractiveAutoencoder(nn.Module):
    """Dense autoencoder with a contractive penalty on encoder activations."""

    def __init__(self, latent_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.flatten = nn.Flatten()

        self.encoder_linears = nn.ModuleList()
        prev_dim = 28 * 28
        for hidden_dim in hidden_dims:
            layer = nn.Linear(prev_dim, hidden_dim)
            self.encoder_linears.append(layer)
            prev_dim = hidden_dim
        self.latent_layer = nn.Linear(prev_dim, latent_dim)

        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, 28 * 28))
        decoder_layers.append(nn.Tanh())
        decoder_layers.append(_Reshape((1, 28, 28)))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self._encode_with_activations(x)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent)

    def _encode_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        activations: list[torch.Tensor] = []
        out = self.flatten(x)
        for layer in self.encoder_linears:
            out = torch.sigmoid(layer(out))
            activations.append(out)
        latent = self.latent_layer(out)
        return latent, activations

    def contractive_penalty(self, x: torch.Tensor) -> torch.Tensor:
        _, activations = self._encode_with_activations(x)
        penalty = torch.tensor(0.0, device=x.device)
        for layer, activation in zip(self.encoder_linears, activations):
            weight = layer.weight
            derivative = activation * (1 - activation)
            frob = torch.sum(weight.pow(2), dim=1)
            penalty = penalty + torch.sum((derivative**2) * frob, dim=1).mean()
        return penalty


class _Reshape(nn.Module):
    def __init__(self, shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return tensor.view(tensor.size(0), *self.shape)
