"""Inference utilities for the wine gradient boosting classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, Field

from .config import CONFIG, GradientBoostingClassificationConfig
from .pipeline import WineGradientBoostingPipeline, train_and_persist


class GradientBoostingClassificationRequest(BaseModel):
    """Typed payload for wine quality predictions."""

    alcohol: float = Field(..., example=13.5)
    malic_acid: float = Field(..., example=1.7)
    ash: float = Field(..., example=2.4)
    alcalinity_of_ash: float = Field(..., example=15.5)
    magnesium: float = Field(..., example=100.0)
    total_phenols: float = Field(..., example=2.9)
    flavanoids: float = Field(..., example=2.3)
    nonflavanoid_phenols: float = Field(..., example=0.3)
    proanthocyanins: float = Field(..., example=1.9)
    color_intensity: float = Field(..., example=5.6)
    hue: float = Field(..., example=1.02)
    od280_od315_of_diluted_wines: float = Field(..., example=3.0)
    proline: float = Field(..., example=1100.0)


class GradientBoostingClassificationResponse(BaseModel):
    """Structured response from the gradient boosting classifier."""

    predicted_label: str
    class_probabilities: dict[str, float]
    model_version: str
    metrics: dict[str, float]
    feature_importances: dict[str, float]


class GradientBoostingClassificationService:
    """High-level service that wraps gradient boosting inference."""

    def __init__(self, config: GradientBoostingClassificationConfig | None = None) -> None:
        self.config = config or CONFIG
        self._pipeline: WineGradientBoostingPipeline | None = None

    def _load_or_train(self) -> WineGradientBoostingPipeline:
        if self._pipeline is not None:
            return self._pipeline
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        pipeline = WineGradientBoostingPipeline(self.config)
        pipeline.pipeline = joblib.load(self.config.model_path)
        self._pipeline = pipeline
        return pipeline

    def _load_metrics(self) -> dict[str, float]:
        if not self.config.metrics_path.exists():
            return {}
        with self.config.metrics_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_feature_importances(self) -> dict[str, float]:
        path = self.config.artifact_dir / "feature_importances.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def predict(self, payload: GradientBoostingClassificationRequest) -> GradientBoostingClassificationResponse:
        pipeline = self._load_or_train()
        model = pipeline.pipeline
        if model is None:
            raise RuntimeError("Pipeline failed to load for inference.")
        row = np.asarray([[getattr(payload, feature) for feature in self.config.feature_columns]])
        prediction = model.predict(row)[0]
        probabilities = model.predict_proba(row)[0]
        class_probabilities = {
            label: float(prob)
            for label, prob in zip(model.classes_, probabilities)
        }
        response = GradientBoostingClassificationResponse(
            predicted_label=str(prediction),
            class_probabilities=class_probabilities,
            model_version=str(int(self.config.model_path.stat().st_mtime)),
            metrics=self._load_metrics(),
            feature_importances=self._load_feature_importances(),
        )
        return response


def get_service(
    config: GradientBoostingClassificationConfig | None = None,
) -> GradientBoostingClassificationService:
    return GradientBoostingClassificationService(config)
