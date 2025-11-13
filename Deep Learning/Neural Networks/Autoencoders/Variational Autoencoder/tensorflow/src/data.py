"""Dataset utilities for the TensorFlow variational autoencoder."""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from .config import VAEConfig


def _prepare(images: tf.Tensor) -> tf.Tensor:
    images = tf.cast(images, tf.float32) / 127.5 - 1.0
    return tf.expand_dims(images, axis=-1)


def load_datasets(config: VAEConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    path = config.data_dir / "fashion-mnist.npz"
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data(path=str(path))

    train_clean = tf.data.Dataset.from_tensor_slices(_prepare(x_train))
    train_clean = train_clean.shuffle(10_000, seed=config.seed, reshuffle_each_iteration=True)
    train_ds = train_clean.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    val_clean = tf.data.Dataset.from_tensor_slices(_prepare(x_test))
    val_ds = val_clean.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
