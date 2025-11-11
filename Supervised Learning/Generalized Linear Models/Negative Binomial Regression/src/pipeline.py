"""Training pipeline for negative binomial regression."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import statsmodels.api as sm

from .config import CONFIG, NegativeBinomialConfig
from .data import train_validation_split


class NegativeBinomialPipeline:
    """Wraps statsmodels GLM workflow for negative binomial counts."""

    def __init__(self, config: NegativeBinomialConfig | None = None) -> None:
        self.config = config or CONFIG
        self.result: sm.GLMResults | None = None

    def build_design_matrices(self, X, y):
        X_matrix = sm.add_constant(X, has_constant="add")
        y_vector = np.asarray(y)
        return X_matrix, y_vector

    def train(self) -> Dict[str, float]:
        X_train, X_val, y_train, y_val = train_validation_split(self.config)
        X_train_matrix, y_train_vector = self.build_design_matrices(X_train, y_train)
        model = sm.GLM(
            y_train_vector,
            X_train_matrix,
            family=sm.families.NegativeBinomial(),
        )
        self.result = model.fit()

        X_val_matrix, y_val_vector = self.build_design_matrices(X_val, y_val)
        predictions = self.result.predict(X_val_matrix)
        predictions = np.clip(predictions, a_min=1e-6, a_max=None)

        deviance = float(self.result.deviance)
        null_deviance = float(self.result.null_deviance)
        pseudo_r2 = 1.0 - deviance / null_deviance if null_deviance else 0.0
        mae = float(np.mean(np.abs(y_val_vector - predictions)))

        metrics: Dict[str, float] = {
            "pseudo_r2": pseudo_r2,
            "mae": mae,
            "theta": float(self.result.params.get("alpha", 0.0)),
        }
        return metrics

    def save(self) -> Path:
        if self.result is None:
            raise RuntimeError("Model is not trained; call train() first.")
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.result.save(self.config.model_path.as_posix())
        return self.config.model_path

    @staticmethod
    def load(path: Path | None = None) -> sm.GLMResults:
        model_path = path or CONFIG.model_path
        return sm.load(model_path.as_posix())

    def write_metrics(self, metrics: Dict[str, float]) -> Path:
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        return self.config.metrics_path


def train_and_persist(
    config: NegativeBinomialConfig | None = None,
) -> Dict[str, float]:
    pipeline = NegativeBinomialPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
