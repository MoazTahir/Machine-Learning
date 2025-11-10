"""Inference utilities for the California housing AdaBoost regressor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from pydantic import BaseModel, Field

from .config import CONFIG, AdaBoostRegressionConfig
from .pipeline import CaliforniaAdaBoostPipeline, train_and_persist


class AdaBoostRegressionRequest(BaseModel):
    """Typed payload for California housing predictions."""

    median_income: float = Field(..., example=5.5)
    house_age: float = Field(..., example=28.0)
    average_rooms: float = Field(..., example=5.7)
    average_bedrooms: float = Field(..., example=1.1)
    population: float = Field(..., example=1200.0)
    average_occupancy: float = Field(..., example=3.1)
    latitude: float = Field(..., example=34.2)
    longitude: float = Field(..., example=-118.4)


class AdaBoostRegressionResponse(BaseModel):
    """Structured response returned by the AdaBoost regressor."""

    predicted_value: float
    model_version: str
    metrics: dict[str, float]
    feature_importances: dict[str, float]
    learning_curve: list[dict[str, float]] | None


class AdaBoostRegressionService:
    """High-level service for AdaBoost regression inference."""

    def __init__(self, config: AdaBoostRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self._pipeline: CaliforniaAdaBoostPipeline | None = None

    def _load_or_train(self) -> CaliforniaAdaBoostPipeline:
        if self._pipeline is not None:
            return self._pipeline
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        pipeline = CaliforniaAdaBoostPipeline(self.config)
        pipeline.pipeline = joblib.load(self.config.model_path)
        self._pipeline = pipeline
        return pipeline

    def _load_metrics(self) -> dict[str, float]:
        if not self.config.metrics_path.exists():
            return {}
        with self.config.metrics_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_feature_importances(self) -> dict[str, float]:
        if not self.config.feature_importances_path.exists():
            return {}
        with self.config.feature_importances_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_learning_curve(self) -> list[dict[str, float]] | None:
        if not self.config.staged_metrics_path.exists():
            return None
        with self.config.staged_metrics_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def predict(self, payload: AdaBoostRegressionRequest) -> AdaBoostRegressionResponse:
        pipeline = self._load_or_train()
        model = pipeline.pipeline
        if model is None:
            raise RuntimeError("Pipeline failed to load for inference.")
        row = np.asarray([[getattr(payload, feature) for feature in self.config.feature_columns]])
        prediction = float(model.predict(row)[0])
        response = AdaBoostRegressionResponse(
            predicted_value=prediction,
            model_version=str(int(self.config.model_path.stat().st_mtime)),
            metrics=self._load_metrics(),
            feature_importances=self._load_feature_importances(),
            learning_curve=self._load_learning_curve(),
        )
        return response


def get_service(
    config: AdaBoostRegressionConfig | None = None,
) -> AdaBoostRegressionService:
    return AdaBoostRegressionService(config)
