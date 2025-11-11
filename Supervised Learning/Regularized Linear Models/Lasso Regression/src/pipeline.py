"""Training pipeline for lasso regression."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, LassoRegressionConfig
from .data import train_validation_split


class LassoRegressionPipeline:
    """Encapsulates preprocessing, lasso fitting, and persistence."""

    def __init__(self, config: LassoRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline: Pipeline | None = None

    def _build_preprocessing(self) -> ColumnTransformer:
        numeric_features = list(self.config.feature_columns)
        numeric_pipeline = Pipeline([("scaler", StandardScaler())])
        return ColumnTransformer([("numeric", numeric_pipeline, numeric_features)])

    def build(self) -> Pipeline:
        preprocessor = self._build_preprocessing()
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    Lasso(
                        alpha=self.config.alpha,
                        max_iter=self.config.max_iter,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )
        return model

    def train(self) -> Dict[str, float]:
        X_train, X_val, y_train, y_val = train_validation_split(self.config)
        self.pipeline = self.build()
        self.pipeline.fit(X_train, y_train)

        predictions = np.asarray(self.pipeline.predict(X_val))
        metrics: Dict[str, float] = {
            "r2": float(r2_score(y_val, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, predictions))),
            "mae": float(mean_absolute_error(y_val, predictions)),
            "sparsity_ratio": float(self._sparsity_ratio()),
        }
        return metrics

    def _sparsity_ratio(self) -> float:
        if self.pipeline is None:
            return 0.0
        model = self.pipeline.named_steps["regressor"]
        coefficients = getattr(model, "coef_", None)
        if coefficients is None:
            return 0.0
        zero_count = float(np.sum(np.isclose(coefficients, 0.0)))
        return zero_count / float(coefficients.size)

    def save(self) -> Path:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not trained; call train() before save().")
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.config.model_path)
        return self.config.model_path

    @staticmethod
    def load(path: Path | None = None) -> Pipeline:
        pipeline_path = path or CONFIG.model_path
        return joblib.load(pipeline_path)

    def write_metrics(self, metrics: Dict[str, float]) -> Path:
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        return self.config.metrics_path


def train_and_persist(
    config: LassoRegressionConfig | None = None,
) -> Dict[str, float]:
    pipeline = LassoRegressionPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
