"""Inference helpers for the TensorFlow denoising autoencoder."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import tensorflow as tf

from .config import CONFIG, DenoisingConfig
from .model import build_model


def load_model(
    config: DenoisingConfig = CONFIG,
    checkpoint_path: Path | None = None,
) -> tf.keras.Model:
    path = checkpoint_path or (config.artifact_dir / "denoising_autoencoder.keras")
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Train the model first with train.py."
        )

    model = build_model(latent_dim=config.latent_dim, hidden_dims=config.hidden_dims)
    loaded = tf.keras.models.load_model(path)
    if isinstance(loaded, tf.keras.Model):
        return loaded
    return model


def denoise(
    images: Iterable[np.ndarray],
    model: tf.keras.Model | None = None,
    config: DenoisingConfig = CONFIG,
) -> List[np.ndarray]:
    if model is None:
        model = load_model(config=config)
    batch = np.stack(list(images), axis=0)
    predictions = model.predict(batch, verbose=0)
    return [pred for pred in predictions]
