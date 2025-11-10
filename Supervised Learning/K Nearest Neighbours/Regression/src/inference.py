"""Inference utilities for the diabetes KNN regressor."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, KNNRegressionConfig
from .pipeline import DiabetesKNNPipeline, train_and_persist


class DiabetesRegressionRequest(BaseModel):
    """Input schema matching the diabetes feature space."""

    age: float = Field(..., description="Normalised age feature")
    sex: float = Field(..., description="Normalised sex feature")
    bmi: float = Field(..., description="Body mass index")
    bp: float = Field(..., description="Average blood pressure")
    s1: float = Field(..., description="TC (total serum cholesterol)")
    s2: float = Field(..., description="LDL (low-density lipoproteins)")
    s3: float = Field(..., description="HDL (high-density lipoproteins)")
    s4: float = Field(..., description="TCH (thyroid stimulating hormone)")
    s5: float = Field(..., description="LTG (lamotrigine level proxy)")
    s6: float = Field(..., description="GLU (blood sugar level)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 0.0381,
                "sex": 0.0507,
                "bmi": 0.0617,
                "bp": 0.0219,
                "s1": -0.0442,
                "s2": -0.0348,
                "s3": -0.0434,
                "s4": -0.0026,
                "s5": 0.0199,
                "s6": -0.0176,
            }
        }
    )


class DiabetesRegressionResponse(BaseModel):
    """Prediction payload served by the FastAPI endpoint."""

    predicted_value: float
    model_version: str
    metrics: dict[str, float]

    model_config = ConfigDict(use_enum_values=True)


class DiabetesRegressionService:
    """High-level service object for diabetes progression predictions."""

    def __init__(self, config: KNNRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self):
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return DiabetesKNNPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: DiabetesRegressionRequest) -> DiabetesRegressionResponse:
        record = {column: getattr(payload, column) for column in self.config.feature_columns}
        data = pd.DataFrame([record])
        predicted_value = float(self.pipeline.predict(data)[0])
        model_version = self._artifact_version(self.config.model_path)
        return DiabetesRegressionResponse(
            predicted_value=predicted_value,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> DiabetesRegressionService:
    return DiabetesRegressionService()


RequestModel = DiabetesRegressionRequest
ResponseModel = DiabetesRegressionResponse
