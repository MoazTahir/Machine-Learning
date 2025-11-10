"""Training utilities for the wine XGBoost classifier."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from .config import CONFIG, XGBoostClassificationConfig
from .data import train_validation_split


class WineXGBoostPipeline:
    """Compose XGBoost training, evaluation, and persistence."""

    def __init__(self, config: XGBoostClassificationConfig | None = None) -> None:
        self.config = config or CONFIG
        self.model: xgb.XGBClassifier | None = None
        self.encoder: LabelEncoder | None = None

    def build(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=len(self.config.class_names),
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
        X_train, X_val, y_train_raw, y_val_raw = train_validation_split(self.config)
        encoder = LabelEncoder()
        encoder.fit(list(self.config.class_names))
        y_train = encoder.transform(y_train_raw)
        y_val = encoder.transform(y_val_raw)

        model = self.build()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        preds_encoded = model.predict(X_val)
        preds = encoder.inverse_transform(preds_encoded.astype(int))
        proba = model.predict_proba(X_val)

        accuracy = float(accuracy_score(y_val_raw, preds))
        macro_precision = float(precision_score(y_val_raw, preds, average="macro", zero_division=0))
        macro_recall = float(recall_score(y_val_raw, preds, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_val_raw, preds, average="macro", zero_division=0))
        loss = float(log_loss(y_val, proba))

        lb = LabelBinarizer()
        lb.fit(list(self.config.class_names))
        y_val_encoded = lb.transform(y_val_raw)
        roc_auc = float(roc_auc_score(y_val_encoded, proba, average="macro", multi_class="ovr"))

        feature_importances = model.feature_importances_.tolist()
        self._write_feature_importances(feature_importances)

        self.model = model
        self.encoder = encoder

        metrics = {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "log_loss": loss,
            "roc_auc": roc_auc,
        }
        return metrics

    def save(self) -> Path:
        if self.model is None or self.encoder is None:
            raise RuntimeError("Model not trained; call train() before save().")
        payload = {
            "model": self.model,
            "encoder": self.encoder,
        }
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, self.config.model_path)
        return self.config.model_path

    def write_metrics(self, metrics: dict[str, float]) -> Path:
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        return self.config.metrics_path

    def _write_feature_importances(self, importances: list[float]) -> None:
        payload = {feature: float(score) for feature, score in zip(self.config.feature_columns, importances)}
        with self.config.feature_importances_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @staticmethod
    def load(path: Path | None = None) -> tuple[xgb.XGBClassifier, LabelEncoder]:
        load_path = path or CONFIG.model_path
        payload = joblib.load(load_path)
        return payload["model"], payload["encoder"]


def train_and_persist(
    config: XGBoostClassificationConfig | None = None,
) -> dict[str, float]:
    pipeline = WineXGBoostPipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
