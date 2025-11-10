"""Training utilities for the wine stochastic gradient boosting classifier."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from .config import CONFIG, StochasticGradientBoostingClassificationConfig
from .data import train_validation_split


class WineStochasticGradientBoostingPipeline:
    """Compose stochastic gradient boosting training, evaluation, and persistence."""

    def __init__(self, config: StochasticGradientBoostingClassificationConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline: Pipeline | None = None

    def build(self) -> Pipeline:
        return Pipeline(
            steps=
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    GradientBoostingClassifier(
                        learning_rate=self.config.learning_rate,
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        subsample=self.config.subsample,
                        max_features=self.config.max_features,
                        min_samples_split=self.config.min_samples_split,
                        min_samples_leaf=self.config.min_samples_leaf,
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
        probs = self.pipeline.predict_proba(X_val)

        accuracy = float(accuracy_score(y_val, preds))
        macro_precision = float(precision_score(y_val, preds, average="macro", zero_division=0))
        macro_recall = float(recall_score(y_val, preds, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        loss = float(log_loss(y_val, probs))

        lb = LabelBinarizer()
        lb.fit(list(self.config.class_names))
        y_val_encoded = lb.transform(y_val)
        roc_auc = float(roc_auc_score(y_val_encoded, probs, average="macro", multi_class="ovr"))

        model: GradientBoostingClassifier = self.pipeline.named_steps["model"]
        feature_importances = model.feature_importances_.tolist()
        self._write_feature_importances(feature_importances)

        metrics: dict[str, float] = {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "log_loss": loss,
            "roc_auc": roc_auc,
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
    config: StochasticGradientBoostingClassificationConfig | None = None,
) -> dict[str, float]:
    pipeline = WineStochasticGradientBoostingPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
