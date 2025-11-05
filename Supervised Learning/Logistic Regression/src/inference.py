"""Inference utilities for the heart disease logistic regression model."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, LogisticRegressionConfig
from .pipeline import HeartDiseasePipeline, train_and_persist


class LogisticRegressionRequest(BaseModel):
    """Input schema mirroring the heart dataset features."""

    age: float = Field(..., ge=0, description="Patient age in years")
    sex: int = Field(..., ge=0, le=1, description="Biological sex (1=male, 0=female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=yes)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=0, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1=yes)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels coloured by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (3 = normal, 6 = fixed defect, 7 = reversable defect)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 54,
                "sex": 1,
                "cp": 1,
                "trestbps": 130,
                "chol": 246,
                "fbs": 0,
                "restecg": 1,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 1,
                "ca": 0,
                "thal": 2,
            }
        }
    )


class LogisticRegressionResponse(BaseModel):
    """Response returned by inference calls."""

    predicted_class: int
    probability: float
    model_version: str
    metrics: dict[str, float]

    model_config = ConfigDict(use_enum_values=True)


class LogisticRegressionService:
    """High-level service wrapper exposing prediction functionality."""

    def __init__(self, config: LogisticRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self):
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return HeartDiseasePipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: LogisticRegressionRequest) -> LogisticRegressionResponse:
        features = np.array([[getattr(payload, field) for field in self.config.feature_columns]], dtype=float)
        proba = float(self.pipeline.predict_proba(features)[0][1])
        prediction = int(proba >= 0.5)
        model_version = self._artifact_version(self.config.model_path)
        return LogisticRegressionResponse(
            predicted_class=prediction,
            probability=proba,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> LogisticRegressionService:
    """Return a cached service instance for FastAPI integration."""
    return LogisticRegressionService()


RequestModel = LogisticRegressionRequest
ResponseModel = LogisticRegressionResponse
