"""Inference utilities for agglomerative clustering."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, AgglomerativeConfig
from .pipeline import AgglomerativePipeline, train_and_persist


class AgglomerativeRequest(BaseModel):
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"x": 5.1, "y": 5.0},
        }
    )


class AgglomerativeResponse(BaseModel):
    cluster_id: int
    model_version: str
    metrics: dict[str, float]


class AgglomerativeService:
    def __init__(self, config: AgglomerativeConfig | None = None) -> None:
        self.config = config or CONFIG
        self.model = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return joblib.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: AgglomerativeRequest) -> AgglomerativeResponse:
        features = np.array([[payload.x, payload.y]], dtype=float)
        cluster_id = int(self.model.fit_predict(features)[0])
        model_version = self._artifact_version(self.config.model_path)
        return AgglomerativeResponse(
            cluster_id=cluster_id,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> AgglomerativeService:
    return AgglomerativeService()


RequestModel = AgglomerativeRequest
ResponseModel = AgglomerativeResponse
