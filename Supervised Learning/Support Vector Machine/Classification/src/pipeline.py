"""Model training pipeline utilities for the SVM classifier."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CONFIG, SVMConfig
from .data import train_validation_split


class BreastCancerSVMPipeline:
    """Encapsulates preprocessing, model training, and persistence."""

    def __init__(self, config: SVMConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline: Pipeline | None = None

    def build(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
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
        proba = np.asarray(self.pipeline.predict_proba(X_val))[:, 1]
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_val, preds)),
            "precision": float(precision_score(y_val, preds)),
            "recall": float(recall_score(y_val, preds)),
            "f1": float(f1_score(y_val, preds)),
            "roc_auc": float(roc_auc_score(y_val, proba)),
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


def train_and_persist(config: SVMConfig | None = None) -> dict[str, float]:
    pipeline = BreastCancerSVMPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
