"""Model architecture for the TensorFlow contractive autoencoder."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import tensorflow as tf

from .utils import PSNRMetric


class ContractiveAutoencoder(tf.keras.Model):
    """Dense autoencoder with an analytic contractive penalty."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: Sequence[int] = (256, 128),
        contractive_weight: float = 1e-3,
    ) -> None:
        super().__init__(name="contractive_autoencoder")
        self.flatten = tf.keras.layers.Flatten()
        self.encoder_linears = [
            tf.keras.layers.Dense(hidden_dim, activation=None) for hidden_dim in hidden_dims
        ]
        self.latent_layer = tf.keras.layers.Dense(latent_dim, activation=None, name="latent")

        self.decoder_layers = [
            tf.keras.layers.Dense(hidden_dim, activation="relu") for hidden_dim in reversed(hidden_dims)
        ]
        self.output_layer = tf.keras.layers.Dense(28 * 28, activation="tanh")
        self.reshape = tf.keras.layers.Reshape((28, 28, 1))

        self.contractive_weight = contractive_weight

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.contractive_tracker = tf.keras.metrics.Mean(name="contractive_penalty")
        self.psnr_metric = PSNRMetric()

    def encode(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.flatten(inputs)
        for layer in self.encoder_linears:
            x = tf.nn.sigmoid(layer(x))
        return self.latent_layer(x)

    def _encode_with_activations(self, inputs: tf.Tensor) -> tuple[tf.Tensor, list[tf.Tensor]]:
        x = self.flatten(inputs)
        activations: list[tf.Tensor] = []
        for layer in self.encoder_linears:
            x = tf.nn.sigmoid(layer(x))
            activations.append(x)
        latent = self.latent_layer(x)
        return latent, activations

    def decode(self, latent: tf.Tensor) -> tf.Tensor:
        x = latent
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.output_layer(x)
        return self.reshape(x)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        return self.decode(latent)

    def contractive_penalty(self, inputs: tf.Tensor) -> tf.Tensor:
        _, activations = self._encode_with_activations(inputs)
        penalty = tf.constant(0.0, dtype=inputs.dtype)
        for layer, activation in zip(self.encoder_linears, activations):
            weight = layer.kernel
            derivative = activation * (1.0 - activation)
            weight_norm = tf.reduce_sum(tf.square(weight), axis=0)
            penalty += tf.reduce_mean(tf.reduce_sum(tf.square(derivative) * weight_norm, axis=1))
        return penalty

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:  # type: ignore[override]
        return [self.loss_tracker, self.recon_tracker, self.contractive_tracker, self.psnr_metric]

    def train_step(self, data: Any) -> Dict[str, tf.Tensor]:  # type: ignore[override]
        inputs, targets = data
        with tf.GradientTape() as tape:
            recon = self(inputs, training=True)
            recon_loss = tf.reduce_mean(tf.square(targets - recon))
            contractive = self.contractive_penalty(inputs)
            loss = recon_loss + self.contractive_weight * contractive

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))  # type: ignore[attr-defined]

        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon_loss)
        self.contractive_tracker.update_state(contractive)
        self.psnr_metric.update_state(targets, recon)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> Dict[str, tf.Tensor]:  # type: ignore[override]
        inputs, targets = data
        recon = self(inputs, training=False)
        recon_loss = tf.reduce_mean(tf.square(targets - recon))
        contractive = self.contractive_penalty(inputs)
        loss = recon_loss + self.contractive_weight * contractive

        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon_loss)
        self.contractive_tracker.update_state(contractive)
        self.psnr_metric.update_state(targets, recon)
        return {metric.name: metric.result() for metric in self.metrics}


def build_model(
    latent_dim: int = 64,
    hidden_dims: Sequence[int] = (256, 128),
    contractive_weight: float = 1e-3,
) -> ContractiveAutoencoder:
    return ContractiveAutoencoder(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        contractive_weight=contractive_weight,
    )
