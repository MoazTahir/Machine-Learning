"""Training utilities for the California housing XGBoost regressor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import CONFIG, XGBoostRegressionConfig
from .data import train_validation_split


class CaliforniaXGBoostPipeline:
    """Compose XGBoost regression training, evaluation, and persistence."""

    def __init__(self, config: XGBoostRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.model: xgb.XGBRegressor | None = None

    def build(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=self.config.reg_lambda,
            reg_alpha=self.config.reg_alpha,
            gamma=self.config.gamma,
            random_state=self.config.random_state,
        )

    def train(self) -> dict[str, float]:
        X_train, X_val, y_train, y_val = train_validation_split(self.config)
        model = self.build()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict(X_val)
        r2 = float(r2_score(y_val, preds))
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))

        self._write_feature_importances(model)
        self._write_learning_curve(model)

        self.model = model

        metrics = {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
        }
        return metrics

    def save(self) -> Path:
        if self.model is None:
            raise RuntimeError("Model not trained; call train() before save().")
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.config.model_path)
        return self.config.model_path

    def write_metrics(self, metrics: dict[str, float]) -> Path:
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        return self.config.metrics_path

    def _write_feature_importances(self, model: xgb.XGBRegressor) -> None:
        payload = {feature: float(score) for feature, score in zip(self.config.feature_columns, model.feature_importances_)}
        with self.config.feature_importances_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _write_learning_curve(self, model: xgb.XGBRegressor) -> None:
        results = getattr(model, "evals_result_", None)
        if not results:
            return
        rmse_values = results.get("validation_0", {}).get("rmse")
        if not rmse_values:
            return
        entries = [{"iteration": idx + 1, "rmse": float(value)} for idx, value in enumerate(rmse_values)]
        with self.config.learning_curve_path.open("w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2)

    @staticmethod
    def load(path: Path | None = None) -> xgb.XGBRegressor:
        load_path = path or CONFIG.model_path
        return joblib.load(load_path)


def train_and_persist(
    config: XGBoostRegressionConfig | None = None,
) -> dict[str, float]:
    pipeline = CaliforniaXGBoostPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
