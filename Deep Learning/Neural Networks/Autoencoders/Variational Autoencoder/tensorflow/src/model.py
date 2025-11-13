"""Model architecture for the TensorFlow variational autoencoder."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import tensorflow as tf

from .utils import PSNRMetric, kl_divergence


class VariationalAutoencoder(tf.keras.Model):
    """Dense VAE with reparameterisation trick."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: Sequence[int] = (256, 128),
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__(name="variational_autoencoder")
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.flatten = tf.keras.layers.Flatten()

        self.encoder_layers = [
            tf.keras.layers.Dense(hidden_dim, activation="relu") for hidden_dim in hidden_dims
        ]
        self.fc_mu = tf.keras.layers.Dense(latent_dim, activation=None)
        self.fc_logvar = tf.keras.layers.Dense(latent_dim, activation=None)

        self.decoder_layers = [
            tf.keras.layers.Dense(hidden_dim, activation="relu") for hidden_dim in reversed(hidden_dims)
        ]
        self.output_layer = tf.keras.layers.Dense(28 * 28, activation="tanh")
        self.reshape = tf.keras.layers.Reshape((28, 28, 1))

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_divergence")
        self.psnr_metric = PSNRMetric()

    def encode(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = self.flatten(inputs)
        for layer in self.encoder_layers:
            x = layer(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        eps = tf.random.normal(shape=tf.shape(mu))
        std = tf.exp(0.5 * logvar)
        return mu + eps * std

    def decode(self, latent: tf.Tensor) -> tf.Tensor:
        x = latent
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.output_layer(x)
        return self.reshape(x)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:  # type: ignore[override]
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:  # type: ignore[override]
        return [self.loss_tracker, self.recon_tracker, self.kl_tracker, self.psnr_metric]

    def train_step(self, data: Any) -> Dict[str, tf.Tensor]:  # type: ignore[override]
        inputs, targets = data
        with tf.GradientTape() as tape:
            mu, logvar = self.encode(inputs)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            recon_loss = tf.reduce_mean(tf.square(targets - recon))
            kl = kl_divergence(mu, logvar)
            loss = recon_loss + self.kl_weight * kl

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))  # type: ignore[attr-defined]

        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon_loss)
        self.kl_tracker.update_state(kl)
        self.psnr_metric.update_state(targets, recon)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> Dict[str, tf.Tensor]:  # type: ignore[override]
        inputs, targets = data
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon_loss = tf.reduce_mean(tf.square(targets - recon))
        kl = kl_divergence(mu, logvar)
        loss = recon_loss + self.kl_weight * kl

        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon_loss)
        self.kl_tracker.update_state(kl)
        self.psnr_metric.update_state(targets, recon)
        return {metric.name: metric.result() for metric in self.metrics}

    def sample(self, num_samples: int) -> tf.Tensor:
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        return self.decode(z)


def build_model(
    latent_dim: int = 32,
    hidden_dims: Sequence[int] = (256, 128),
    kl_weight: float = 1.0,
) -> VariationalAutoencoder:
    return VariationalAutoencoder(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        kl_weight=kl_weight,
    )
