"""Agglomerative clustering pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from .config import CONFIG, AgglomerativeConfig
from .data import load_dataset


class AgglomerativePipeline:
    """Wraps clustering, evaluation, and persistence for hierarchical clustering."""

    def __init__(self, config: AgglomerativeConfig | None = None) -> None:
        self.config = config or CONFIG
        self.model: AgglomerativeClustering | None = None

    def train(self) -> Dict[str, float]:
        df = load_dataset(self.config)
        features = df[list(self.config.feature_columns)].to_numpy()

        self.model = AgglomerativeClustering(
            n_clusters=self.config.n_clusters,
            linkage=self.config.linkage,
        )
        labels = self.model.fit_predict(features)
        metrics: Dict[str, float] = {
            "silhouette_score": float(silhouette_score(features, labels)),
        }
        return metrics

    def save(self) -> Path:
        if self.model is None:
            raise RuntimeError("Model is not trained; call train() before save().")
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.config.model_path)
        return self.config.model_path

    def write_report(self, metrics: Dict[str, float]) -> Path:
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        return self.config.metrics_path


def train_and_persist(config: AgglomerativeConfig | None = None) -> Dict[str, float]:
    pipeline = AgglomerativePipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_report(metrics)
    return metrics
