"""Inference utilities for isotonic calibration."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, IsotonicRegressionConfig
from .pipeline import IsotonicRegressionPipeline, train_and_persist


class IsotonicRegressionRequest(BaseModel):
    feature_1: float = Field(..., description="Primary feature value")
    feature_2: float = Field(..., description="Secondary feature value")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feature_1": 1.8,
                "feature_2": 4.0,
            }
        }
    )


class IsotonicRegressionResponse(BaseModel):
    calibrated_probability: float
    model_version: str
    metrics: dict[str, float]


class IsotonicRegressionService:
    def __init__(self, config: IsotonicRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return IsotonicRegressionPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: IsotonicRegressionRequest) -> IsotonicRegressionResponse:
        features = np.array([[payload.feature_1, payload.feature_2]], dtype=float)
        probability = float(self.pipeline.predict_proba(features)[0, 1])
        model_version = self._artifact_version(self.config.model_path)
        return IsotonicRegressionResponse(
            calibrated_probability=probability,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> IsotonicRegressionService:
    return IsotonicRegressionService()


RequestModel = IsotonicRegressionRequest
ResponseModel = IsotonicRegressionResponse
