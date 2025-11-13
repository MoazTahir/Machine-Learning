"""Training entry point for the PyTorch contractive autoencoder."""
from __future__ import annotations

from typing import Dict

from .config import CONFIG, ContractiveConfig
from .engine import train_contractive_autoencoder
from .model import ContractiveAutoencoder
from .utils import save_metrics, set_seed


def train(config: ContractiveConfig = CONFIG) -> Dict[str, float]:
    set_seed(config.seed)
    config.ensure_dirs()

    model = ContractiveAutoencoder(latent_dim=config.latent_dim, hidden_dims=config.hidden_dims)
    model.to(config.device)

    history = train_contractive_autoencoder(config, model)

    metrics = {
        "train_loss": history["train_loss"][-1],
        "train_psnr": history["train_psnr"][-1],
        "train_contractive": history["train_contractive"][-1],
        "val_loss": history["val_loss"][-1],
        "val_psnr": history["val_psnr"][-1],
        "val_contractive": history["val_contractive"][-1],
    }
    save_metrics(metrics, config.artifact_dir / "metrics.json")
    return metrics


if __name__ == "__main__":
    results = train(CONFIG)
    print(results)
