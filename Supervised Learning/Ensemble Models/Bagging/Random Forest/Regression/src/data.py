"""Data utilities for the California housing random forest regressor."""
from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from .config import CONFIG, RandomForestRegressionConfig


def _normalise_column(name: str) -> str:
    cleaned = (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _fetch_dataset(config: RandomForestRegressionConfig) -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame.copy()
    rename: Callable[[str], str] = _normalise_column
    frame = frame.rename(columns={col: rename(col) for col in frame.columns})
    frame = frame.rename(columns={"target": config.target_column})
    frame[config.target_column] = frame[config.target_column].astype(float)
    config.data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.data_path, index=False)
    return frame


def load_dataset(config: RandomForestRegressionConfig = CONFIG) -> pd.DataFrame:
    """Return a cached copy of the dataset."""
    if config.data_path.exists():
        return pd.read_csv(config.data_path)
    return _fetch_dataset(config)


def build_features(
    df: pd.DataFrame, config: RandomForestRegressionConfig = CONFIG
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and continuous target."""
    X = df[list(config.feature_columns)]
    y = df[config.target_column].astype(float)
    return X, y


def train_validation_split(
    config: RandomForestRegressionConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return train/validation sets with stratified income quantiles."""
    df = load_dataset(config)
    X, y = build_features(df, config)
    stratify_bins = pd.qcut(y, q=10, duplicates="drop")
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_bins,
    )
