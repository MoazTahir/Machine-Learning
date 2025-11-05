"""Model training pipeline utilities for logistic regression."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, LogisticRegressionConfig
from .data import train_validation_split


class HeartDiseasePipeline:
    """Encapsulates preprocessing, model training, and persistence."""

    def __init__(self, config: LogisticRegressionConfig | None = None) -> None:
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
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )
        return model

    def train(self) -> dict[str, float]:
        """Train the pipeline and return evaluation metrics."""
        X_train, X_val, y_train, y_val = train_validation_split(self.config)
        self.pipeline = self.build()
        self.pipeline.fit(X_train, y_train)

        predictions = np.asarray(self.pipeline.predict(X_val))
        probabilities = np.asarray(self.pipeline.predict_proba(X_val))[:, 1]
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_val, predictions)),
            "precision": float(precision_score(y_val, predictions)),
            "recall": float(recall_score(y_val, predictions)),
            "f1": float(f1_score(y_val, predictions)),
            "roc_auc": float(roc_auc_score(y_val, probabilities)),
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

    def write_metrics(self, metrics: dict[str, float]) -> Path:
        """Write evaluation metrics to disk for later inspection."""
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        return self.config.metrics_path


def train_and_persist(
    config: LogisticRegressionConfig | None = None,
) -> dict[str, float]:
    """Train the pipeline, persist artifacts, and return metrics."""
    pipeline = HeartDiseasePipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
