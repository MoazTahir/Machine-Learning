"""Model definition for the PyTorch variational autoencoder."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class VariationalAutoencoder(nn.Module):
    """MLP-based VAE for 28x28 grayscale images."""

    def __init__(self, latent_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        input_dim = 28 * 28

        encoder_layers: list[nn.Module] = [nn.Flatten()]
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.fc_mu.out_features, device=device)
        return self.decode(z)


class _Reshape(nn.Module):
    def __init__(self, shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return tensor.view(tensor.size(0), *self.shape)
