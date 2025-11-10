"""Data utilities for the breast cancer AdaBoost classifier."""
from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from .config import CONFIG, AdaBoostClassificationConfig


def _normalise_column(name: str) -> str:
    cleaned = (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "_")
    )
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _fetch_dataset(config: AdaBoostClassificationConfig) -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    rename: Callable[[str], str] = _normalise_column
    frame = frame.rename(columns={col: rename(col) for col in frame.columns})
    frame = frame.rename(columns={"target": config.target_column})
    class_mapping = {0: "malignant", 1: "benign"}
    frame[config.target_column] = frame[config.target_column].map(class_mapping)
    config.data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.data_path, index=False)
    return frame


def load_dataset(config: AdaBoostClassificationConfig = CONFIG) -> pd.DataFrame:
    if config.data_path.exists():
        return pd.read_csv(config.data_path)
    return _fetch_dataset(config)


def build_features(
    df: pd.DataFrame, config: AdaBoostClassificationConfig = CONFIG
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[list(config.feature_columns)]
    y = df[config.target_column].astype(str)
    return X, y


def train_validation_split(
    config: AdaBoostClassificationConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_dataset(config)
    X, y = build_features(df, config)
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
