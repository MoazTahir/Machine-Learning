"""Inference utilities for the California housing random forest regressor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from pydantic import BaseModel, Field

from .config import CONFIG, RandomForestRegressionConfig
from .pipeline import CaliforniaRandomForestPipeline, train_and_persist


class RandomForestRegressionRequest(BaseModel):
    """Typed payload for California housing inference."""

    median_income: float = Field(..., example=5.4321)
    house_age: float = Field(..., example=28.0)
    average_rooms: float = Field(..., example=5.8)
    average_bedrooms: float = Field(..., example=1.1)
    population: float = Field(..., example=1400.0)
    average_occupancy: float = Field(..., example=3.0)
    latitude: float = Field(..., example=34.21)
    longitude: float = Field(..., example=-118.45)


class RandomForestRegressionResponse(BaseModel):
    """Structured response returned by the random forest regressor."""

    predicted_value: float
    model_version: str
    metrics: dict[str, float]
    feature_importances: dict[str, float]


class RandomForestRegressionService:
    """High-level service for random forest housing predictions."""

    def __init__(self, config: RandomForestRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self._pipeline: CaliforniaRandomForestPipeline | None = None

    def _load_or_train(self) -> CaliforniaRandomForestPipeline:
        if self._pipeline is not None:
            return self._pipeline
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        pipeline = CaliforniaRandomForestPipeline(self.config)
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

    def predict(self, payload: RandomForestRegressionRequest) -> RandomForestRegressionResponse:
        pipeline = self._load_or_train()
        model = pipeline.pipeline
        if model is None:
            raise RuntimeError("Pipeline failed to load for inference.")
        row = np.asarray([[getattr(payload, feature) for feature in self.config.feature_columns]])
        prediction = float(model.predict(row)[0])
        response = RandomForestRegressionResponse(
            predicted_value=prediction,
            model_version=str(int(self.config.model_path.stat().st_mtime)),
            metrics=self._load_metrics(),
            feature_importances=self._load_feature_importances(),
        )
        return response


def get_service(config: RandomForestRegressionConfig | None = None) -> RandomForestRegressionService:
    return RandomForestRegressionService(config)
