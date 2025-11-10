"""Training pipeline utilities for the breast cancer random forest classifier."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, RandomForestClassificationConfig
from .data import train_validation_split


class BreastCancerRandomForestPipeline:
    """Compose random forest training, evaluation, and persistence."""

    def __init__(self, config: RandomForestClassificationConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline: Pipeline | None = None

    def build(self) -> Pipeline:
        return Pipeline(
            steps=
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        min_samples_split=self.config.min_samples_split,
                        min_samples_leaf=self.config.min_samples_leaf,
                        max_features=self.config.max_features,
                        bootstrap=self.config.bootstrap,
                        class_weight=self.config.class_weight,
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
        accuracy = float(accuracy_score(y_val, preds))
        macro_precision = float(precision_score(y_val, preds, average="macro", zero_division=0))
        macro_recall = float(recall_score(y_val, preds, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

        model: RandomForestClassifier = self.pipeline.named_steps["model"]
        class_list = list(model.classes_)
        positive_idx = class_list.index(self.config.positive_label)
        proba = self.pipeline.predict_proba(X_val)[:, positive_idx]
        roc_auc = float(
            roc_auc_score([1 if label == self.config.positive_label else 0 for label in y_val], proba)
        )

        feature_importances = model.feature_importances_.tolist()

        metrics: dict[str, float] = {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "roc_auc": roc_auc,
            "num_trees": float(model.n_estimators),
            "max_depth": float(model.max_depth or -1),
        }
        self._write_feature_importances(feature_importances, class_list)
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

    def _write_feature_importances(self, importances: list[float], class_list: list[str]) -> None:
        payload = {
            "feature_importances": dict(zip(self.config.feature_columns, importances)),
            "classes": class_list,
        }
        path = self.config.artifact_dir / "feature_importances.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @staticmethod
    def load(path: Path | None = None) -> Pipeline:
        pipeline_path = path or CONFIG.model_path
        return joblib.load(pipeline_path)


def train_and_persist(
    config: RandomForestClassificationConfig | None = None,
) -> dict[str, float]:
    pipeline = BreastCancerRandomForestPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
