"""Inference service for elastic net regression."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, ElasticNetConfig
from .pipeline import ElasticNetPipeline, train_and_persist


class ElasticNetRequest(BaseModel):
    experience: float = Field(..., ge=0.0, description="Years of experience")
    feature_group1: float = Field(..., description="First correlated feature group")
    feature_group2: float = Field(..., description="Second correlated feature group")
    feature_noise: float = Field(..., description="Noise feature")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experience": 6.5,
                "feature_group1": 6.2,
                "feature_group2": 6.1,
                "feature_noise": 0.03,
            }
        }
    )


class ElasticNetResponse(BaseModel):
    salary_prediction: float
    model_version: str
    metrics: dict[str, float]


class ElasticNetService:
    def __init__(self, config: ElasticNetConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return ElasticNetPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: ElasticNetRequest) -> ElasticNetResponse:
        features = np.array(
            [
                [
                    payload.experience,
                    payload.feature_group1,
                    payload.feature_group2,
                    payload.feature_noise,
                ]
            ],
            dtype=float,
        )
        prediction = float(self.pipeline.predict(features)[0])
        model_version = self._artifact_version(self.config.model_path)
        return ElasticNetResponse(
            salary_prediction=prediction,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> ElasticNetService:
    return ElasticNetService()


RequestModel = ElasticNetRequest
ResponseModel = ElasticNetResponse
