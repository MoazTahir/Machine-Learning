"""Inference helpers for the TensorFlow variational autoencoder."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import tensorflow as tf

from .config import CONFIG, VAEConfig
from .model import build_model
from .utils import PSNRMetric


def load_model(
    config: VAEConfig = CONFIG,
    checkpoint_path: Path | None = None,
) -> tf.keras.Model:
    path = checkpoint_path or (config.artifact_dir / "variational_autoencoder.weights.h5")
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Train the model first with train.py."
        )

    model = build_model(
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        kl_weight=config.kl_weight,
    )
    model.psnr_metric = PSNRMetric()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate))
    model.load_weights(str(path))
    return model


def reconstruct(
    images: Iterable[np.ndarray],
    model: tf.keras.Model | None = None,
    config: VAEConfig = CONFIG,
) -> List[np.ndarray]:
    if model is None:
        model = load_model(config=config)
    batch = np.stack(list(images), axis=0)
    outputs = model.predict(batch, verbose=0)
    return [pred for pred in outputs]


def sample(
    num_samples: int,
    model: tf.keras.Model | None = None,
    config: VAEConfig = CONFIG,
) -> np.ndarray:
    if model is None:
        model = load_model(config=config)
    if hasattr(model, "sample"):
        return model.sample(num_samples).numpy()
    fresh = build_model(
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        kl_weight=config.kl_weight,
    )
    fresh.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate))
    path = config.artifact_dir / "variational_autoencoder.weights.h5"
    if path.exists():
        fresh.load_weights(str(path))
    return fresh.sample(num_samples).numpy()
