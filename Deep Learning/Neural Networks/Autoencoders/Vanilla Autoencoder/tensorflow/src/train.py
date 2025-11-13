"""Training script for the TensorFlow vanilla autoencoder."""
from __future__ import annotations

import json
from typing import Dict

import tensorflow as tf

from .config import CONFIG, AutoencoderConfig
from .data import load_datasets
from .model import build_model
from .utils import compile_model, save_history


def set_seed(seed: int) -> None:
    tf.random.set_seed(seed)


def train(config: AutoencoderConfig = CONFIG) -> Dict[str, float]:
    config.ensure_dirs()
    set_seed(config.seed)

    train_ds, val_ds = load_datasets(config)
    model = build_model(latent_dim=64, hidden_dims=(256, 128))
    model = compile_model(model, learning_rate=config.learning_rate)

    checkpoint_path = config.artifact_dir / "vanilla_autoencoder.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.num_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    val_loss = history.history.get("val_loss", [0.0])
    val_psnr = history.history.get("val_psnr_metric", [0.0])

    summary = {
        "best_val_loss": float(min(val_loss)),
        "best_val_psnr": float(max(val_psnr)),
        "final_val_loss": float(val_loss[-1]),
        "final_val_psnr": float(val_psnr[-1]),
    }

    save_history(history, summary, config.artifact_dir / "metrics.json")
    return summary


if __name__ == "__main__":
    metrics = train()
    print(json.dumps(metrics, indent=2))
