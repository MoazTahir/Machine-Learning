"""Inference utilities for the breast cancer SVM classifier."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, SVMConfig
from .pipeline import BreastCancerSVMPipeline, train_and_persist


class BreastCancerRequest(BaseModel):
    """Input schema mirroring the breast cancer feature space."""

    mean_radius: float = Field(..., ge=0)
    mean_texture: float = Field(..., ge=0)
    mean_perimeter: float = Field(..., ge=0)
    mean_area: float = Field(..., ge=0)
    mean_smoothness: float = Field(..., ge=0)
    mean_compactness: float = Field(..., ge=0)
    mean_concavity: float = Field(..., ge=0)
    mean_concave_points: float = Field(..., ge=0)
    mean_symmetry: float = Field(..., ge=0)
    mean_fractal_dimension: float = Field(..., ge=0)
    radius_error: float = Field(..., ge=0)
    texture_error: float = Field(..., ge=0)
    perimeter_error: float = Field(..., ge=0)
    area_error: float = Field(..., ge=0)
    smoothness_error: float = Field(..., ge=0)
    compactness_error: float = Field(..., ge=0)
    concavity_error: float = Field(..., ge=0)
    concave_points_error: float = Field(..., ge=0)
    symmetry_error: float = Field(..., ge=0)
    fractal_dimension_error: float = Field(..., ge=0)
    worst_radius: float = Field(..., ge=0)
    worst_texture: float = Field(..., ge=0)
    worst_perimeter: float = Field(..., ge=0)
    worst_area: float = Field(..., ge=0)
    worst_smoothness: float = Field(..., ge=0)
    worst_compactness: float = Field(..., ge=0)
    worst_concavity: float = Field(..., ge=0)
    worst_concave_points: float = Field(..., ge=0)
    worst_symmetry: float = Field(..., ge=0)
    worst_fractal_dimension: float = Field(..., ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mean_radius": 20.57,
                "mean_texture": 17.77,
                "mean_perimeter": 132.90,
                "mean_area": 1326.0,
                "mean_smoothness": 0.08474,
                "mean_compactness": 0.07864,
                "mean_concavity": 0.08690,
                "mean_concave_points": 0.07017,
                "mean_symmetry": 0.1812,
                "mean_fractal_dimension": 0.05667,
                "radius_error": 1.095,
                "texture_error": 0.9053,
                "perimeter_error": 8.589,
                "area_error": 153.4,
                "smoothness_error": 0.006399,
                "compactness_error": 0.04904,
                "concavity_error": 0.05373,
                "concave_points_error": 0.01587,
                "symmetry_error": 0.03003,
                "fractal_dimension_error": 0.006193,
                "worst_radius": 25.38,
                "worst_texture": 17.33,
                "worst_perimeter": 184.60,
                "worst_area": 2019.0,
                "worst_smoothness": 0.1622,
                "worst_compactness": 0.6656,
                "worst_concavity": 0.7119,
                "worst_concave_points": 0.2654,
                "worst_symmetry": 0.4601,
                "worst_fractal_dimension": 0.1189,
            }
        }
    )


class BreastCancerResponse(BaseModel):
    """Prediction payload served by the FastAPI endpoint."""

    predicted_label: str
    probability_malignant: float
    model_version: str
    metrics: dict[str, float]

    model_config = ConfigDict(use_enum_values=True)


class BreastCancerService:
    """High-level service object for malignancy predictions."""

    def __init__(self, config: SVMConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self):
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return BreastCancerSVMPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: BreastCancerRequest) -> BreastCancerResponse:
        data = pd.DataFrame(
            [
                {column: getattr(payload, column) for column in self.config.feature_columns}
            ]
        )
        proba_malignant = float(self.pipeline.predict_proba(data)[0][1])
        label = self.config.positive_label if proba_malignant >= 0.5 else "benign"
        model_version = self._artifact_version(self.config.model_path)
        return BreastCancerResponse(
            predicted_label=label,
            probability_malignant=proba_malignant,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> BreastCancerService:
    return BreastCancerService()


RequestModel = BreastCancerRequest
ResponseModel = BreastCancerResponse
