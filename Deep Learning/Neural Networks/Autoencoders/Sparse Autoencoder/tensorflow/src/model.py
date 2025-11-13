"""Model architecture for the TensorFlow sparse autoencoder."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import tensorflow as tf

from .utils import PSNRMetric, kl_divergence


class SparseAutoencoder(tf.keras.Model):
    """Dense autoencoder with a KL sparsity constraint on the latent code."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: Sequence[int] = (256, 128),
        sparsity_target: float = 0.05,
        sparsity_weight: float = 1e-3,
    ) -> None:
        super().__init__(name="sparse_autoencoder")
        self.flatten = tf.keras.layers.Flatten()
        self.encoder_layers = [
            tf.keras.layers.Dense(hidden_dim, activation="sigmoid") for hidden_dim in hidden_dims
        ]
        self.latent_layer = tf.keras.layers.Dense(latent_dim, activation="sigmoid", name="latent")

        self.decoder_layers = [
            tf.keras.layers.Dense(hidden_dim, activation="relu") for hidden_dim in reversed(hidden_dims)
        ]
        self.output_layer = tf.keras.layers.Dense(28 * 28, activation="tanh")
        self.reshape = tf.keras.layers.Reshape((28, 28, 1))

        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_penalty")
        self.psnr_metric = PSNRMetric()

    def encode(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.flatten(inputs)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.latent_layer(x)

    def decode(self, latent: tf.Tensor) -> tf.Tensor:
        x = latent
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.output_layer(x)
        return self.reshape(x)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        return self.decode(latent)

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:  # type: ignore[override]
        return [self.loss_tracker, self.recon_tracker, self.kl_tracker, self.psnr_metric]

    def train_step(self, data: Any) -> Dict[str, tf.Tensor]:  # type: ignore[override]
        inputs, targets = data
        with tf.GradientTape() as tape:
            recon = self(inputs, training=True)
            recon_loss = tf.reduce_mean(tf.square(targets - recon))
            kl_penalty = kl_divergence(self.sparsity_target, self.encode(inputs))
            loss = recon_loss + self.sparsity_weight * kl_penalty

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))  # type: ignore[attr-defined]

        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon_loss)
        self.kl_tracker.update_state(kl_penalty)
        self.psnr_metric.update_state(targets, recon)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> Dict[str, tf.Tensor]:  # type: ignore[override]
        inputs, targets = data
        recon = self(inputs, training=False)
        recon_loss = tf.reduce_mean(tf.square(targets - recon))
        kl_penalty = kl_divergence(self.sparsity_target, self.encode(inputs))
        loss = recon_loss + self.sparsity_weight * kl_penalty

        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon_loss)
        self.kl_tracker.update_state(kl_penalty)
        self.psnr_metric.update_state(targets, recon)
        return {metric.name: metric.result() for metric in self.metrics}


def build_model(
    latent_dim: int = 64,
    hidden_dims: Sequence[int] = (256, 128),
    sparsity_target: float = 0.05,
    sparsity_weight: float = 1e-3,
) -> SparseAutoencoder:
    return SparseAutoencoder(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        sparsity_target=sparsity_target,
        sparsity_weight=sparsity_weight,
    )
