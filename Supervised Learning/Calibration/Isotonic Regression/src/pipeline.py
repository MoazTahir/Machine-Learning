"""Training pipeline for isotonic regression calibration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, IsotonicRegressionConfig
from .data import train_validation_split


def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    indices = np.clip(np.floor(probs * bins).astype(int), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = indices == b
        if not np.any(mask):
            continue
        avg_confidence = probs[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += np.abs(avg_accuracy - avg_confidence) * mask.mean()
    return float(ece)


class IsotonicRegressionPipeline:
    """Pipeline combining base estimator with isotonic calibration."""

    def __init__(self, config: IsotonicRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline: Pipeline | None = None

    def build(self) -> Pipeline:
        scaler = StandardScaler()
        base_model = LogisticRegression(C=self.config.base_estimator_C, max_iter=1000)
        calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
        pipeline = Pipeline(
            [
                ("scaler", scaler),
                ("calibrator", calibrated),
            ]
        )
        return pipeline

    def train(self) -> Dict[str, float]:
        X_train, X_val, y_train, y_val = train_validation_split(self.config)
        self.pipeline = self.build()
        self.pipeline.fit(X_train, y_train)

        probs = np.asarray(self.pipeline.predict_proba(X_val)[:, 1])
        metrics: Dict[str, float] = {
            "brier_score": float(brier_score_loss(y_val, probs)),
            "expected_calibration_error": _expected_calibration_error(
                y_val.to_numpy(dtype=float), probs
            ),
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
        model_path = path or CONFIG.model_path
        return joblib.load(model_path)

    def write_metrics(self, metrics: Dict[str, float]) -> Path:
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        return self.config.metrics_path


def train_and_persist(config: IsotonicRegressionConfig | None = None) -> Dict[str, float]:
    pipeline = IsotonicRegressionPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
