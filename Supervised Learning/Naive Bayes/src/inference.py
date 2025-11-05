"""Inference utilities for the mushroom Naive Bayes classifier."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, NaiveBayesConfig
from .pipeline import MushroomPipeline, train_and_persist


class NaiveBayesRequest(BaseModel):
    """Input schema mirroring mushroom categorical features."""

    cap_shape: str = Field(..., alias="cap-shape")
    cap_surface: str = Field(..., alias="cap-surface")
    cap_color: str = Field(..., alias="cap-color")
    bruises: str
    odor: str
    gill_attachment: str = Field(..., alias="gill-attachment")
    gill_spacing: str = Field(..., alias="gill-spacing")
    gill_size: str = Field(..., alias="gill-size")
    gill_color: str = Field(..., alias="gill-color")
    stalk_shape: str = Field(..., alias="stalk-shape")
    stalk_root: str | None = Field(None, alias="stalk-root")
    stalk_surface_above_ring: str = Field(..., alias="stalk-surface-above-ring")
    stalk_surface_below_ring: str = Field(..., alias="stalk-surface-below-ring")
    stalk_color_above_ring: str = Field(..., alias="stalk-color-above-ring")
    stalk_color_below_ring: str = Field(..., alias="stalk-color-below-ring")
    veil_type: str = Field(..., alias="veil-type")
    veil_color: str = Field(..., alias="veil-color")
    ring_number: str = Field(..., alias="ring-number")
    ring_type: str = Field(..., alias="ring-type")
    spore_print_color: str = Field(..., alias="spore-print-color")
    population: str
    habitat: str

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "cap-shape": "x",
                "cap-surface": "s",
                "cap-color": "n",
                "bruises": "t",
                "odor": "a",
                "gill-attachment": "f",
                "gill-spacing": "c",
                "gill-size": "b",
                "gill-color": "k",
                "stalk-shape": "e",
                "stalk-root": "b",
                "stalk-surface-above-ring": "s",
                "stalk-surface-below-ring": "s",
                "stalk-color-above-ring": "w",
                "stalk-color-below-ring": "w",
                "veil-type": "p",
                "veil-color": "w",
                "ring-number": "o",
                "ring-type": "p",
                "spore-print-color": "k",
                "population": "s",
                "habitat": "u",
            }
        },
    )


class NaiveBayesResponse(BaseModel):
    """Prediction payload served over the FastAPI endpoint."""

    predicted_label: str
    probability_poisonous: float
    model_version: str
    metrics: dict[str, float]

    model_config = ConfigDict(use_enum_values=True)


class NaiveBayesService:
    """High-level service object for mushroom edibility predictions."""

    def __init__(self, config: NaiveBayesConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return MushroomPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: NaiveBayesRequest) -> NaiveBayesResponse:
        data = pd.DataFrame([
            {
                column: getattr(payload, column.replace("-", "_"))
                for column in self.config.feature_columns
            }
        ])
        data = data.replace({pd.NA: None})
        proba_poisonous = float(self.pipeline.predict_proba(data)[0][1])
        label = "poisonous" if proba_poisonous >= 0.5 else "edible"
        model_version = self._artifact_version(self.config.model_path)
        return NaiveBayesResponse(
            predicted_label=label,
            probability_poisonous=proba_poisonous,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> NaiveBayesService:
    return NaiveBayesService()


RequestModel = NaiveBayesRequest
ResponseModel = NaiveBayesResponse
