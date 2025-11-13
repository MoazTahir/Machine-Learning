"""Utilities for the TensorFlow sparse autoencoder."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf


def kl_divergence(rho: float, activations: tf.Tensor) -> tf.Tensor:
    rho_hat = tf.reduce_mean(activations, axis=0)
    rho_hat = tf.clip_by_value(rho_hat, 1e-7, 1 - 1e-7)
    term1 = rho * tf.math.log(rho / rho_hat)
    term2 = (1 - rho) * tf.math.log((1 - rho) / (1 - rho_hat))
    return tf.reduce_sum(term1 + term2)


class PSNRMetric(tf.keras.metrics.Mean):
    """Average PSNR for tensors normalised to [-1, 1]."""

    def __init__(self, name: str = "psnr_metric", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor | None = None) -> None:  # type: ignore[override]
        psnr_values = tf.image.psnr(y_true, y_pred, max_val=2.0)
        super().update_state(psnr_values, sample_weight=sample_weight)


def save_history(history: tf.keras.callbacks.History, summary: Dict[str, Any], path: Path) -> None:
    payload = {
        "history": {k: [float(v) for v in values] for k, values in history.history.items()},
        "summary": summary,
    }
    path.write_text(json.dumps(payload, indent=2))
