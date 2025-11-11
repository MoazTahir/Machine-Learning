"""Inference helpers for lasso regression."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, LassoRegressionConfig
from .pipeline import LassoRegressionPipeline, train_and_persist


class LassoRegressionRequest(BaseModel):
    experience: float = Field(..., ge=0.0, description="Years of experience")
    feature_sparse1: float = Field(..., description="Sparse feature 1")
    feature_sparse2: float = Field(..., description="Sparse feature 2")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experience": 6.5,
                "feature_sparse1": 0.05,
                "feature_sparse2": -0.02,
            }
        }
    )


class LassoRegressionResponse(BaseModel):
    salary_prediction: float
    model_version: str
    metrics: dict[str, float]


class LassoRegressionService:
    def __init__(self, config: LassoRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return LassoRegressionPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: LassoRegressionRequest) -> LassoRegressionResponse:
        features = np.array(
            [[payload.experience, payload.feature_sparse1, payload.feature_sparse2]],
            dtype=float,
        )
        prediction = float(self.pipeline.predict(features)[0])
        model_version = self._artifact_version(self.config.model_path)
        return LassoRegressionResponse(
            salary_prediction=prediction,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> LassoRegressionService:
    return LassoRegressionService()


RequestModel = LassoRegressionRequest
ResponseModel = LassoRegressionResponse
