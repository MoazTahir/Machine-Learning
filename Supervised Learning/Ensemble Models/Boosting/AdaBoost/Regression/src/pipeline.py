"""Training utilities for the California housing AdaBoost regressor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .config import CONFIG, AdaBoostRegressionConfig
from .data import train_validation_split


class CaliforniaAdaBoostPipeline:
    """Compose AdaBoost regression training, evaluation, and persistence."""

    def __init__(self, config: AdaBoostRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline: Pipeline | None = None

    def build(self) -> Pipeline:
        estimator = DecisionTreeRegressor(
            max_depth=self.config.estimator_max_depth,
            random_state=self.config.random_state,
        )
        return Pipeline(
            steps=
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    AdaBoostRegressor(
                        estimator=estimator,
                        n_estimators=self.config.n_estimators,
                        learning_rate=self.config.learning_rate,
                        loss=self.config.loss,
                        random_state=self.config.random_state,
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

        model: AdaBoostRegressor = self.pipeline.named_steps["model"]
        self._write_feature_importances(model)
        self._write_learning_curve(model, X_val, y_val)

        metrics = {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
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

    def _write_feature_importances(self, model: AdaBoostRegressor) -> None:
        payload = {feature: float(score) for feature, score in zip(self.config.feature_columns, model.feature_importances_)}
        with self.config.feature_importances_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _write_learning_curve(self, model: AdaBoostRegressor, X_val, y_val) -> None:
        staged_entries: list[dict[str, float]] = []
        for iteration, prediction in enumerate(model.staged_predict(X_val), start=1):
            rmse = float(np.sqrt(mean_squared_error(y_val, prediction)))
            staged_entries.append({"iteration": iteration, "rmse": rmse})
        if not staged_entries:
            return
        with self.config.staged_metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(staged_entries, fh, indent=2)

    @staticmethod
    def load(path: Path | None = None) -> Pipeline:
        pipeline_path = path or CONFIG.model_path
        return joblib.load(pipeline_path)


def train_and_persist(
    config: AdaBoostRegressionConfig | None = None,
) -> dict[str, float]:
    pipeline = CaliforniaAdaBoostPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
