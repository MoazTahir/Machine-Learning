"""Inference utilities for the California housing SVR model."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, SVRConfig
from .pipeline import CaliforniaHousingSVRPipeline, train_and_persist


class CaliforniaHousingRequest(BaseModel):
    """Feature schema mirroring the SVR training data."""

    median_income: float = Field(..., ge=0, description="Median income in block (scaled by 10k)")
    house_age: float = Field(..., ge=0, description="Median house age in block")
    average_rooms: float = Field(..., ge=0, description="Average number of rooms per household")
    average_bedrooms: float = Field(..., ge=0, description="Average number of bedrooms per household")
    population: float = Field(..., ge=0, description="Block population")
    average_occupancy: float = Field(..., ge=0, description="Average household occupancy")
    latitude: float = Field(..., description="Latitude of the block group")
    longitude: float = Field(..., description="Longitude of the block group")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "median_income": 8.3252,
                "house_age": 41.0,
                "average_rooms": 6.9841,
                "average_bedrooms": 1.0238,
                "population": 322.0,
                "average_occupancy": 2.5556,
                "latitude": 37.88,
                "longitude": -122.23,
            }
        }
    )


class CaliforniaHousingResponse(BaseModel):
    """Prediction payload for the SVR FastAPI endpoint."""

    predicted_value: float
    model_version: str
    metrics: dict[str, float]

    model_config = ConfigDict(use_enum_values=True)


class CaliforniaHousingService:
    """High-level service object for SVR predictions."""

    def __init__(self, config: SVRConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self):
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return CaliforniaHousingSVRPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: CaliforniaHousingRequest) -> CaliforniaHousingResponse:
        frame = pd.DataFrame(
            [
                {column: getattr(payload, column) for column in self.config.feature_columns}
            ]
        )
        prediction = float(self.pipeline.predict(frame)[0])
        model_version = self._artifact_version(self.config.model_path)
        return CaliforniaHousingResponse(
            predicted_value=prediction,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> CaliforniaHousingService:
    return CaliforniaHousingService()


RequestModel = CaliforniaHousingRequest
ResponseModel = CaliforniaHousingResponse
