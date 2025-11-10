"""Training pipeline utilities for the California housing random forest regressor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, RandomForestRegressionConfig
from .data import train_validation_split


class CaliforniaRandomForestPipeline:
    """Compose training, evaluation, and persistence for the regressor."""

    def __init__(self, config: RandomForestRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline: Pipeline | None = None

    def build(self) -> Pipeline:
        return Pipeline(
            steps=
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        min_samples_split=self.config.min_samples_split,
                        min_samples_leaf=self.config.min_samples_leaf,
                        max_features=self.config.max_features,
                        bootstrap=self.config.bootstrap,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def train(self) -> dict[str, float]:
        X_train, X_val, y_train, y_val = train_validation_split(self.config)
        self.pipeline = self.build()
        self.pipeline.fit(X_train, y_train)

        preds = np.asarray(self.pipeline.predict(X_val))
        r2 = float(r2_score(y_val, preds))
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))

        model: RandomForestRegressor = self.pipeline.named_steps["model"]
        feature_importances = model.feature_importances_.tolist()
        self._write_feature_importances(feature_importances)

        metrics = {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "num_trees": float(model.n_estimators),
            "max_depth": float(model.max_depth or -1),
        }
        return metrics

    def save(self) -> Path:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not trained; call train() before save().")
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.config.model_path)
        return self.config.model_path

    def write_metrics(self, metrics: dict[str, float]) -> Path:
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        return self.config.metrics_path

    def _write_feature_importances(self, importances: list[float]) -> None:
        payload = {feature: float(score) for feature, score in zip(self.config.feature_columns, importances)}
        path = self.config.artifact_dir / "feature_importances.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @staticmethod
    def load(path: Path | None = None) -> Pipeline:
        pipeline_path = path or CONFIG.model_path
        return joblib.load(pipeline_path)


def train_and_persist(config: RandomForestRegressionConfig | None = None) -> dict[str, float]:
    pipeline = CaliforniaRandomForestPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
