"""Inference utilities for the wine XGBoost classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder

from .config import CONFIG, XGBoostClassificationConfig
from .data import build_features, load_dataset
from .pipeline import WineXGBoostPipeline, train_and_persist


class XGBoostClassificationRequest(BaseModel):
    """Typed payload for wine classification predictions."""

    alcohol: float = Field(..., example=13.2)
    malic_acid: float = Field(..., example=1.78)
    ash: float = Field(..., example=2.14)
    alcalinity_of_ash: float = Field(..., example=11.2)
    magnesium: float = Field(..., example=100.0)
    total_phenols: float = Field(..., example=2.65)
    flavanoids: float = Field(..., example=2.76)
    nonflavanoid_phenols: float = Field(..., example=0.26)
    proanthocyanins: float = Field(..., example=1.28)
    color_intensity: float = Field(..., example=4.38)
    hue: float = Field(..., example=1.05)
    od280_od315_of_diluted_wines: float = Field(..., example=3.4)
    proline: float = Field(..., example=1050.0)


class XGBoostClassificationResponse(BaseModel):
    """Structured response returned by the XGBoost classifier."""

    predicted_label: str
    class_probabilities: dict[str, float]
    model_version: str
    metrics: dict[str, float]
    feature_importances: dict[str, float]


class XGBoostClassificationService:
    """High-level service for XGBoost wine classification inference."""

    def __init__(self, config: XGBoostClassificationConfig | None = None) -> None:
        self.config = config or CONFIG
        self._model: xgb.XGBClassifier | None = None
        self._encoder: LabelEncoder | None = None

    def _ensure_model(self) -> tuple[xgb.XGBClassifier, LabelEncoder]:
        if self._model is not None and self._encoder is not None:
            return self._model, self._encoder
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        payload: dict[str, Any] = joblib.load(self.config.model_path)
        model = payload["model"]
        encoder = payload["encoder"]
        self._model = model
        self._encoder = encoder
        return model, encoder

    def _load_metrics(self) -> dict[str, float]:
        if not self.config.metrics_path.exists():
            return {}
        with self.config.metrics_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_feature_importances(self) -> dict[str, float]:
        if not self.config.feature_importances_path.exists():
            return {}
        with self.config.feature_importances_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def predict(self, payload: XGBoostClassificationRequest) -> XGBoostClassificationResponse:
        model, encoder = self._ensure_model()
        row = np.asarray([[getattr(payload, feature) for feature in self.config.feature_columns]])
        probabilities = model.predict_proba(row)[0]
        predicted_idx = int(np.argmax(probabilities))
        predicted_label = encoder.inverse_transform([predicted_idx])[0]
        class_probabilities = {label: float(prob) for label, prob in zip(encoder.classes_, probabilities)}
        response = XGBoostClassificationResponse(
            predicted_label=str(predicted_label),
            class_probabilities=class_probabilities,
            model_version=str(int(self.config.model_path.stat().st_mtime)),
            metrics=self._load_metrics(),
            feature_importances=self._load_feature_importances(),
        )
        return response


def get_service(
    config: XGBoostClassificationConfig | None = None,
) -> XGBoostClassificationService:
    return XGBoostClassificationService(config)
