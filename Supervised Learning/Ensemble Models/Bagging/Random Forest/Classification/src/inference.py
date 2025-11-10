"""Inference utilities for the breast cancer random forest classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, Field

from .config import CONFIG, RandomForestClassificationConfig
from .pipeline import BreastCancerRandomForestPipeline, train_and_persist


class RandomForestClassificationRequest(BaseModel):
    """Typed payload for breast cancer predictions."""

    mean_radius: float = Field(..., example=17.99)
    mean_texture: float = Field(..., example=10.38)
    mean_perimeter: float = Field(..., example=122.8)
    mean_area: float = Field(..., example=1001.0)
    mean_smoothness: float = Field(..., example=0.1184)
    mean_compactness: float = Field(..., example=0.2776)
    mean_concavity: float = Field(..., example=0.3001)
    mean_concave_points: float = Field(..., example=0.1471)
    mean_symmetry: float = Field(..., example=0.2419)
    mean_fractal_dimension: float = Field(..., example=0.07871)
    radius_error: float = Field(..., example=1.095)
    texture_error: float = Field(..., example=0.9053)
    perimeter_error: float = Field(..., example=8.589)
    area_error: float = Field(..., example=153.4)
    smoothness_error: float = Field(..., example=0.006399)
    compactness_error: float = Field(..., example=0.04904)
    concavity_error: float = Field(..., example=0.05373)
    concave_points_error: float = Field(..., example=0.01587)
    symmetry_error: float = Field(..., example=0.03003)
    fractal_dimension_error: float = Field(..., example=0.006193)
    worst_radius: float = Field(..., example=25.38)
    worst_texture: float = Field(..., example=17.33)
    worst_perimeter: float = Field(..., example=184.6)
    worst_area: float = Field(..., example=2019.0)
    worst_smoothness: float = Field(..., example=0.1622)
    worst_compactness: float = Field(..., example=0.6656)
    worst_concavity: float = Field(..., example=0.7119)
    worst_concave_points: float = Field(..., example=0.2654)
    worst_symmetry: float = Field(..., example=0.4601)
    worst_fractal_dimension: float = Field(..., example=0.1189)


class RandomForestClassificationResponse(BaseModel):
    """Structured response from the random forest classifier."""

    predicted_label: str
    probability_of_malignant: float
    model_version: str
    metrics: dict[str, float]
    feature_importances: dict[str, float]


class RandomForestClassificationService:
    """High-level service for random forest breast cancer inference."""

    def __init__(self, config: RandomForestClassificationConfig | None = None) -> None:
        self.config = config or CONFIG
        self._pipeline: BreastCancerRandomForestPipeline | None = None

    def _load_or_train(self) -> BreastCancerRandomForestPipeline:
        if self._pipeline is not None:
            return self._pipeline
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        pipeline = BreastCancerRandomForestPipeline(self.config)
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
            payload = json.load(fh)
        return payload.get("feature_importances", {})

    def predict(self, payload: RandomForestClassificationRequest) -> RandomForestClassificationResponse:
        pipeline = self._load_or_train()
        model = pipeline.pipeline
        if model is None:
            raise RuntimeError("Pipeline failed to load for inference.")
        data = np.asarray([[getattr(payload, field) for field in self.config.feature_columns]])
        prediction = model.predict(data)[0]
        proba = model.predict_proba(data)[0]
        class_list = list(model.classes_)
        positive_idx = class_list.index(self.config.positive_label)
        response = RandomForestClassificationResponse(
            predicted_label=str(prediction),
            probability_of_malignant=float(proba[positive_idx]),
            model_version=str(int(self.config.model_path.stat().st_mtime)),
            metrics=self._load_metrics(),
            feature_importances=self._load_feature_importances(),
        )
        return response


def get_service(config: RandomForestClassificationConfig | None = None) -> RandomForestClassificationService:
    return RandomForestClassificationService(config)
