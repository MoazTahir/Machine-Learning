"""Inference helpers for the TensorFlow contractive autoencoder."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import tensorflow as tf

from .config import CONFIG, ContractiveConfig
from .model import build_model
from .utils import PSNRMetric


def load_model(
    config: ContractiveConfig = CONFIG,
    checkpoint_path: Path | None = None,
) -> tf.keras.Model:
    path = checkpoint_path or (config.artifact_dir / "contractive_autoencoder.weights.h5")
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Train the model first with train.py."
        )

    model = build_model(
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        contractive_weight=config.contractive_weight,
    )
    model.psnr_metric = PSNRMetric()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate))
    model.load_weights(str(path))
    return model


def reconstruct(
    images: Iterable[np.ndarray],
    model: tf.keras.Model | None = None,
    config: ContractiveConfig = CONFIG,
) -> List[np.ndarray]:
    if model is None:
        model = load_model(config=config)
    batch = np.stack(list(images), axis=0)
    outputs = model.predict(batch, verbose=0)
    return [pred for pred in outputs]
