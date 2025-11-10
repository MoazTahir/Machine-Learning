"""Data utilities for the California housing stochastic gradient boosting regressor."""
from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from .config import CONFIG, StochasticGradientBoostingRegressionConfig


def _normalise(name: str) -> str:
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


def _fetch_dataset(config: StochasticGradientBoostingRegressionConfig) -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame.copy()
    normalise: Callable[[str], str] = _normalise
    frame = frame.rename(columns={col: normalise(col) for col in frame.columns})
    frame = frame.rename(columns={"target": config.target_column})
    config.data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.data_path, index=False)
    return frame


def load_dataset(config: StochasticGradientBoostingRegressionConfig = CONFIG) -> pd.DataFrame:
    if config.data_path.exists():
        return pd.read_csv(config.data_path)
    return _fetch_dataset(config)


def build_features(
    df: pd.DataFrame, config: StochasticGradientBoostingRegressionConfig = CONFIG
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[list(config.feature_columns)]
    y = df[config.target_column].astype(float)
    return X, y


def train_validation_split(
    config: StochasticGradientBoostingRegressionConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
