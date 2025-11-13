"""Training entry point for the PyTorch denoising autoencoder."""
from __future__ import annotations

from typing import Dict

from .config import CONFIG, DenoisingConfig
from .engine import train_denoising_autoencoder
from .model import DenoisingAutoencoder
from .utils import save_metrics, set_seed


def train(config: DenoisingConfig = CONFIG) -> Dict[str, float]:
    set_seed(config.seed)
    config.ensure_dirs()

    model = DenoisingAutoencoder(latent_dim=config.latent_dim, hidden_dims=config.hidden_dims)
    model.to(config.device)

    history = train_denoising_autoencoder(config, model)

    metrics = {
        "train_loss": history["train_loss"][-1],
        "train_psnr": history["train_psnr"][-1],
        "val_loss": history["val_loss"][-1],
        "val_psnr": history["val_psnr"][-1],
    }
    save_metrics(metrics, config.artifact_dir / "metrics.json")
    return metrics


if __name__ == "__main__":
    results = train(CONFIG)
    print(results)
