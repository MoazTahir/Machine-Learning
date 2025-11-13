"""Utilities for the TensorFlow denoising autoencoder."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf


class PSNRMetric(tf.keras.metrics.Mean):
    """Average PSNR for tensors normalised to [-1, 1]."""

    def __init__(self, name: str = "psnr_metric", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor | None = None) -> None:  # type: ignore[override]
        psnr_values = tf.image.psnr(y_true, y_pred, max_val=2.0)
        super().update_state(psnr_values, sample_weight=sample_weight)


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[PSNRMetric()],
    )
    return model


def save_history(history: tf.keras.callbacks.History, summary: Dict[str, Any], path: Path) -> None:
    payload = {
        "history": {k: [float(v) for v in values] for k, values in history.history.items()},
        "summary": summary,
    }
    path.write_text(json.dumps(payload, indent=2))
