"""Model training pipeline utilities for Support Vector Regression."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import CONFIG, SVRConfig
from .data import train_validation_split


class CaliforniaHousingSVRPipeline:
    """Encapsulates preprocessing, SVR training, and persistence."""

    def __init__(self, config: SVRConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline: Pipeline | None = None

    def build(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1),
                ),
            ]
        )

    def train(self) -> dict[str, float]:
        X_train, X_val, y_train, y_val = train_validation_split(self.config)
        self.pipeline = self.build()
        self.pipeline.fit(X_train, y_train)

        predictions = np.asarray(self.pipeline.predict(X_val))
        metrics: dict[str, float] = {
            "r2": float(r2_score(y_val, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, predictions))),
            "mae": float(mean_absolute_error(y_val, predictions)),
        }
        return metrics

    def save(self) -> Path:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not trained; call train() before save().")
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.config.model_path)
        return self.config.model_path

    @staticmethod
    def load(path: Path | None = None) -> Pipeline:
        pipeline_path = path or CONFIG.model_path
        return joblib.load(pipeline_path)

    def write_metrics(self, metrics: dict[str, float]) -> Path:
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        return self.config.metrics_path


def train_and_persist(config: SVRConfig | None = None) -> dict[str, float]:
    pipeline = CaliforniaHousingSVRPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
