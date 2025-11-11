"""Inference service for Poisson regression."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, PoissonRegressionConfig
from .pipeline import PoissonRegressionPipeline, train_and_persist


class PoissonRegressionRequest(BaseModel):
    weekday: int = Field(..., ge=1, le=7, description="Day of week indicator")
    exposure_hours: float = Field(..., ge=0.0, description="Exposure hours")
    promotions: int = Field(..., ge=0, description="Promotion count")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "weekday": 2,
                "exposure_hours": 8.0,
                "promotions": 1,
            }
        }
    )


class PoissonRegressionResponse(BaseModel):
    expected_count: float
    model_version: str
    metrics: dict[str, float]


class PoissonRegressionService:
    def __init__(self, config: PoissonRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return PoissonRegressionPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: PoissonRegressionRequest) -> PoissonRegressionResponse:
        features = np.array(
            [[payload.weekday, payload.exposure_hours, payload.promotions]],
            dtype=float,
        )
        expected_count = float(self.pipeline.predict(features)[0])
        model_version = self._artifact_version(self.config.model_path)
        return PoissonRegressionResponse(
            expected_count=expected_count,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> PoissonRegressionService:
    return PoissonRegressionService()


RequestModel = PoissonRegressionRequest
ResponseModel = PoissonRegressionResponse
