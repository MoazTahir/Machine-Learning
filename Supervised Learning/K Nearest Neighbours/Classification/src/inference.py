"""Inference utilities for the wine KNN classifier."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, KNNClassificationConfig
from .pipeline import WineKNNPipeline, train_and_persist


class WineClassificationRequest(BaseModel):
    """Input schema aligned with the wine feature space."""

    alcohol: float = Field(..., ge=0)
    malic_acid: float = Field(..., ge=0)
    ash: float = Field(..., ge=0)
    alcalinity_of_ash: float = Field(..., ge=0)
    magnesium: float = Field(..., ge=0)
    total_phenols: float = Field(..., ge=0)
    flavanoids: float = Field(..., ge=0)
    nonflavanoid_phenols: float = Field(..., ge=0)
    proanthocyanins: float = Field(..., ge=0)
    color_intensity: float = Field(..., ge=0)
    hue: float = Field(..., ge=0)
    od280_od315_of_diluted_wines: float = Field(..., ge=0)
    proline: float = Field(..., ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "alcohol": 13.05,
                "malic_acid": 1.73,
                "ash": 2.04,
                "alcalinity_of_ash": 12.4,
                "magnesium": 92.0,
                "total_phenols": 2.72,
                "flavanoids": 3.27,
                "nonflavanoid_phenols": 0.17,
                "proanthocyanins": 1.98,
                "color_intensity": 3.0,
                "hue": 1.05,
                "od280_od315_of_diluted_wines": 3.58,
                "proline": 520.0,
            }
        }
    )


class WineClassificationResponse(BaseModel):
    """Prediction payload served by the FastAPI endpoint."""

    predicted_label: str
    class_probabilities: dict[str, float]
    model_version: str
    metrics: dict[str, float]

    model_config = ConfigDict(use_enum_values=True)


class WineClassificationService:
    """High-level service object for wine variety predictions."""

    def __init__(self, config: KNNClassificationConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self):
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return WineKNNPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: WineClassificationRequest) -> WineClassificationResponse:
        record = {column: getattr(payload, column) for column in self.config.feature_columns}
        data = pd.DataFrame([record])
        predicted_label = str(self.pipeline.predict(data)[0])
        probabilities = self.pipeline.predict_proba(data)[0]
        class_probabilities = {
            str(label): float(prob)
            for label, prob in zip(self.pipeline.classes_, probabilities)
        }
        model_version = self._artifact_version(self.config.model_path)
        return WineClassificationResponse(
            predicted_label=predicted_label,
            class_probabilities=class_probabilities,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> WineClassificationService:
    return WineClassificationService()


RequestModel = WineClassificationRequest
ResponseModel = WineClassificationResponse
