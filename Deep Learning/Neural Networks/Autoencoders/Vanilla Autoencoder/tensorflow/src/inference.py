"""Inference helpers for the TensorFlow vanilla autoencoder."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import tensorflow as tf

from .config import CONFIG, AutoencoderConfig
from .model import build_model
from .utils import PSNRMetric


def load_model(
    config: AutoencoderConfig = CONFIG,
    checkpoint_path: Path | None = None,
) -> tf.keras.Model:
    path = checkpoint_path or (config.artifact_dir / "vanilla_autoencoder.keras")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}. Train the model first.")
    return tf.keras.models.load_model(
        str(path),
        custom_objects={"PSNRMetric": PSNRMetric},
    )


def reconstruct(
    images: Iterable[np.ndarray],
    model: tf.keras.Model | None = None,
    config: AutoencoderConfig = CONFIG,
) -> List[np.ndarray]:
    if model is None:
        model = load_model(config=config)
    batch = np.stack([img for img in images])
    predictions = model.predict(batch, verbose=0)
    return [pred for pred in predictions]
