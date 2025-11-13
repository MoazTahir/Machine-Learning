"""Training script for the TensorFlow sparse autoencoder."""
from __future__ import annotations

import json
from typing import Dict

import tensorflow as tf

from .config import CONFIG, SparseConfig
from .data import load_datasets
from .model import build_model
from .utils import PSNRMetric, save_history


def set_seed(seed: int) -> None:
    tf.random.set_seed(seed)


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.psnr_metric = PSNRMetric()
    model.compile(optimizer=optimizer)
    return model


def train(config: SparseConfig = CONFIG) -> Dict[str, float]:
    config.ensure_dirs()
    set_seed(config.seed)

    train_ds, val_ds = load_datasets(config)
    model = build_model(
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        sparsity_target=config.sparsity_target,
        sparsity_weight=config.sparsity_weight,
    )
    model = compile_model(model, learning_rate=config.learning_rate)

    checkpoint_path = config.artifact_dir / "sparse_autoencoder.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.num_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    if checkpoint_path.exists():
        model.load_weights(str(checkpoint_path))

    val_loss = history.history.get("val_loss", [0.0])
    val_psnr = history.history.get("val_psnr_metric", [0.0])
    val_kl = history.history.get("val_kl_penalty", [0.0])

    summary = {
        "best_val_loss": float(min(val_loss)),
        "best_val_psnr": float(max(val_psnr)),
        "best_val_kl": float(min(val_kl)),
        "final_val_loss": float(val_loss[-1]),
        "final_val_psnr": float(val_psnr[-1]),
        "final_val_kl": float(val_kl[-1]),
    }

    save_history(history, summary, config.artifact_dir / "metrics.json")
    return summary


if __name__ == "__main__":
    metrics = train()
    print(json.dumps(metrics, indent=2))
