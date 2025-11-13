"""Model architecture for the TensorFlow vanilla autoencoder."""
from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf


def build_model(latent_dim: int = 64, hidden_dims: Sequence[int] = (256, 128)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    for hidden_dim in hidden_dims:
        x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    latent = tf.keras.layers.Dense(latent_dim, name="latent")(x)

    x = latent
    for hidden_dim in reversed(hidden_dims):
        x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = tf.keras.layers.Dense(28 * 28, activation="tanh")(x)
    outputs = tf.keras.layers.Reshape((28, 28, 1))(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="vanilla_autoencoder")
