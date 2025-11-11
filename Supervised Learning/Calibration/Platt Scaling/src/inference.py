"""Inference utilities for Platt scaling."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, PlattScalingConfig
from .pipeline import PlattScalingPipeline, train_and_persist


class PlattScalingRequest(BaseModel):
    feature_1: float = Field(..., description="Primary feature value")
    feature_2: float = Field(..., description="Secondary feature value")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feature_1": 1.6,
                "feature_2": 3.6,
            }
        }
    )


class PlattScalingResponse(BaseModel):
    calibrated_probability: float
    model_version: str
    metrics: dict[str, float]


class PlattScalingService:
    def __init__(self, config: PlattScalingConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return PlattScalingPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: PlattScalingRequest) -> PlattScalingResponse:
        features = np.array([[payload.feature_1, payload.feature_2]], dtype=float)
        probability = float(self.pipeline.predict_proba(features)[0, 1])
        model_version = self._artifact_version(self.config.model_path)
        return PlattScalingResponse(
            calibrated_probability=probability,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> PlattScalingService:
    return PlattScalingService()


RequestModel = PlattScalingRequest
ResponseModel = PlattScalingResponse
