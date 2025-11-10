"""Data loading utilities for the California housing regression task."""
from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from .config import CONFIG, SVRConfig


def _normalise_column(name: str) -> str:
    cleaned = name.strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _load_from_source(config: SVRConfig) -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    rename_fn: Callable[[str], str] = _normalise_column
    df = df.rename(columns={col: rename_fn(col) for col in df.columns})
    df = df.rename(columns={"medhousevalue": config.target_column})
    config.data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.data_path, index=False)
    return df


def load_dataset(config: SVRConfig = CONFIG) -> pd.DataFrame:
    """Load the dataset from disk (download if necessary)."""
    if config.data_path.exists():
        return pd.read_csv(config.data_path)
    return _load_from_source(config)


def build_features(
    df: pd.DataFrame, config: SVRConfig = CONFIG
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into features and target."""
    X = df[list(config.feature_columns)]
    y = df[config.target_column]
    return X, y


def train_validation_split(
    config: SVRConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a stratified split (by quantiles) for regression stability."""
    df = load_dataset(config)
    df = df.copy()
    df["_bucket"] = pd.qcut(df[config.target_column], q=10, duplicates="drop")
    X, y = build_features(df, config)
    buckets = df.pop("_bucket")
    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        X,
        y,
        buckets,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=buckets,
    )
    return X_train, X_val, y_train, y_val
