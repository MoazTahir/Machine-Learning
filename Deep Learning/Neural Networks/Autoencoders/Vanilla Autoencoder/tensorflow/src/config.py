"""Configuration for the TensorFlow vanilla autoencoder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tensorflow as tf


@dataclass(slots=True, frozen=True)
class AutoencoderConfig:
    data_dir: Path = Path("artifacts/tensorflow_vanilla_ae/data")
    artifact_dir: Path = Path("artifacts/tensorflow_vanilla_ae")
    batch_size: int = 256
    num_epochs: int = 15
    learning_rate: float = 1e-3
    seed: int = 1337
    device_preference: Literal["MPS", "GPU", "CPU"] = "MPS"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


def configure_devices(config: AutoencoderConfig) -> None:
    physical = tf.config.list_physical_devices()
    if config.device_preference == "MPS":
        mps_devices = [d for d in physical if d.device_type.upper() == "GPU" and "MPS" in d.name.upper()]
        if mps_devices:
            tf.config.set_visible_devices(mps_devices, "GPU")
            return
    if config.device_preference in {"MPS", "GPU"}:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices(gpus, "GPU")
            return
    tf.config.set_visible_devices([], "GPU")


CONFIG = AutoencoderConfig()
CONFIG.ensure_dirs()
configure_devices(CONFIG)
