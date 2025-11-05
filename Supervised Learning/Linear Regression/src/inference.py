"""Inference utilities for serving linear regression predictions."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, LinearRegressionConfig
from .pipeline import LinearRegressionPipeline, train_and_persist


class LinearRegressionRequest(BaseModel):
    """Request schema for salary prediction."""

    years_experience: float = Field(..., ge=0.0, description="Years of professional experience")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"years_experience": 5.5},
        }
    )


class LinearRegressionResponse(BaseModel):
    """Response schema returned by inference calls."""

    salary_prediction: float
    model_version: str
    metrics: dict[str, float]

    model_config = ConfigDict(use_enum_values=True)


class LinearRegressionService:
    """High level service object that wraps the trained pipeline."""

    def __init__(self, config: LinearRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return LinearRegressionPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: LinearRegressionRequest) -> LinearRegressionResponse:
        features = np.array([[payload.years_experience]], dtype=float)
        salary = float(self.pipeline.predict(features)[0])
        model_version = self._artifact_version(self.config.model_path)
        return LinearRegressionResponse(
            salary_prediction=salary,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> LinearRegressionService:
    """Factory returning a cached service instance for FastAPI integration."""
    return LinearRegressionService()


RequestModel = LinearRegressionRequest
ResponseModel = LinearRegressionResponse
