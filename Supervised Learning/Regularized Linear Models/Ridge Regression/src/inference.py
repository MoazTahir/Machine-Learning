"""Inference utilities for ridge regression."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, RidgeRegressionConfig
from .pipeline import RidgeRegressionPipeline, train_and_persist


class RidgeRegressionRequest(BaseModel):
    """Request schema for ridge regression inference."""

    experience: float = Field(..., ge=0.0, description="Years of experience")
    feature_multicollinear: float = Field(..., description="Highly correlated feature variant")
    feature_noise: float = Field(..., description="Noise feature capturing random variation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experience": 6.5,
                "feature_multicollinear": 6.2,
                "feature_noise": 0.05,
            }
        }
    )


class RidgeRegressionResponse(BaseModel):
    """Response schema returned by the ridge inference endpoint."""

    salary_prediction: float
    model_version: str
    metrics: dict[str, float]

    model_config = ConfigDict(use_enum_values=True)


class RidgeRegressionService:
    """Service wrapper around the trained ridge pipeline."""

    def __init__(self, config: RidgeRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return RidgeRegressionPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: RidgeRegressionRequest) -> RidgeRegressionResponse:
        features = np.array(
            [
                [
                    payload.experience,
                    payload.feature_multicollinear,
                    payload.feature_noise,
                ]
            ],
            dtype=float,
        )
        prediction = float(self.pipeline.predict(features)[0])
        model_version = self._artifact_version(self.config.model_path)
        return RidgeRegressionResponse(
            salary_prediction=prediction,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> RidgeRegressionService:
    """Factory returning a cached ridge service instance for FastAPI."""

    return RidgeRegressionService()


RequestModel = RidgeRegressionRequest
ResponseModel = RidgeRegressionResponse
