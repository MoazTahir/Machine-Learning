"""Model training pipeline utilities for linear regression."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, LinearRegressionConfig
from .data import train_validation_split


class LinearRegressionPipeline:
    """Encapsulates preprocessing, model training, and persistence."""

    def __init__(self, config: LinearRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline: Pipeline | None = None

    def _build_preprocessing(self) -> ColumnTransformer:
        numeric_features = list(self.config.feature_columns)
        numeric_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
            ],
            remainder="drop",
        )

    def build(self) -> Pipeline:
        """Create a new scikit-learn Pipeline instance."""
        preprocessor = self._build_preprocessing()
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        )
        return model

    def train(self) -> Dict[str, float]:
        """Train the pipeline and return evaluation metrics."""
        X_train, X_val, y_train, y_val = train_validation_split(self.config)
        self.pipeline = self.build()
        self.pipeline.fit(X_train, y_train)

        predictions = np.asarray(self.pipeline.predict(X_val))
        metrics: Dict[str, float] = {
            "r2": float(r2_score(y_val, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, predictions))),
            "mae": float(mean_absolute_error(y_val, predictions)),
        }
        return metrics

    def save(self) -> Path:
        """Persist the trained pipeline to disk."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not trained; call train() before save().")
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.config.model_path)
        return self.config.model_path

    @staticmethod
    def load(path: Path | None = None) -> Pipeline:
        """Load a persisted pipeline from disk."""
        pipeline_path = path or CONFIG.model_path
        return joblib.load(pipeline_path)

    def write_metrics(self, metrics: Dict[str, float]) -> Path:
        """Write evaluation metrics to disk for later inspection."""
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        return self.config.metrics_path


def train_and_persist(config: LinearRegressionConfig | None = None) -> Dict[str, float]:
    """Train the pipeline, persist artifacts, and return metrics."""
    pipeline = LinearRegressionPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
