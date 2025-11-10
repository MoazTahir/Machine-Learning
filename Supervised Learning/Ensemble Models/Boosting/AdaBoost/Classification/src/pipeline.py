"""Training utilities for the breast cancer AdaBoost classifier."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import CONFIG, AdaBoostClassificationConfig
from .data import train_validation_split


class BreastCancerAdaBoostPipeline:
    """Compose AdaBoost training, evaluation, and persistence."""

    def __init__(self, config: AdaBoostClassificationConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline: Pipeline | None = None

    def build(self) -> Pipeline:
        estimator = DecisionTreeClassifier(
            max_depth=self.config.estimator_max_depth,
            random_state=self.config.random_state,
        )
        return Pipeline(
            steps=
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    AdaBoostClassifier(
                        estimator=estimator,
                        n_estimators=self.config.n_estimators,
                        learning_rate=self.config.learning_rate,
                        algorithm=self.config.algorithm,
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
        proba_matrix = self.pipeline.predict_proba(X_val)
        class_list = list(self.config.class_names)
        positive_index = class_list.index(self.config.positive_label)
        positive_proba = proba_matrix[:, positive_index]

        accuracy = float(accuracy_score(y_val, preds))
        precision = float(precision_score(y_val, preds, pos_label=self.config.positive_label, zero_division=0))
        recall = float(recall_score(y_val, preds, pos_label=self.config.positive_label, zero_division=0))
        f1 = float(f1_score(y_val, preds, pos_label=self.config.positive_label, zero_division=0))
        auc = float(
            roc_auc_score([1 if label == self.config.positive_label else 0 for label in y_val], positive_proba)
        )
        loss = float(log_loss(y_val, proba_matrix))

        model: AdaBoostClassifier = self.pipeline.named_steps["model"]
        feature_importances = model.feature_importances_.tolist()
        self._write_feature_importances(feature_importances)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
            "log_loss": loss,
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


def train_and_persist(
    config: AdaBoostClassificationConfig | None = None,
) -> dict[str, float]:
    pipeline = BreastCancerAdaBoostPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
