"""Inference helpers for negative binomial regression."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, NegativeBinomialConfig
from .pipeline import NegativeBinomialPipeline, train_and_persist


class NegativeBinomialRequest(BaseModel):
    weekday: int = Field(..., ge=1, le=7, description="Day of week indicator")
    exposure_hours: float = Field(..., ge=0.0, description="Exposure hours")
    promotions: int = Field(..., ge=0, description="Promotions running")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "weekday": 5,
                "exposure_hours": 8.0,
                "promotions": 2,
            }
        }
    )


class NegativeBinomialResponse(BaseModel):
    expected_count: float
    model_version: str
    metrics: dict[str, float]


class NegativeBinomialService:
    def __init__(self, config: NegativeBinomialConfig | None = None) -> None:
        self.config = config or CONFIG
        self.model = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return NegativeBinomialPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: NegativeBinomialRequest) -> NegativeBinomialResponse:
        features = np.array([[payload.weekday, payload.exposure_hours, payload.promotions]])
        design = sm.add_constant(features, has_constant="add")
        expected_count = float(self.model.predict(design)[0])
        model_version = self._artifact_version(self.config.model_path)
        return NegativeBinomialResponse(
            expected_count=expected_count,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> NegativeBinomialService:
    return NegativeBinomialService()


RequestModel = NegativeBinomialRequest
ResponseModel = NegativeBinomialResponse
