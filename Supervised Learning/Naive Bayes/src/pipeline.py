"""Model training pipeline utilities for the Naive Bayes classifier."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from .config import CONFIG, NaiveBayesConfig
from .data import train_validation_split


class MushroomPipeline:
    """Encapsulates preprocessing, model training, and persistence."""

    def __init__(self, config: NaiveBayesConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline: Pipeline | None = None

    def _build_preprocessing(self) -> ColumnTransformer:
        categorical_features = list(self.config.feature_columns)
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("categorical", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
        )

    def build(self) -> Pipeline:
        preprocessor = self._build_preprocessing()
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", CategoricalNB()),
            ]
        )
        return model

    def train(self) -> dict[str, float]:
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


def train_and_persist(config: NaiveBayesConfig | None = None) -> dict[str, float]:
    pipeline = MushroomPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
