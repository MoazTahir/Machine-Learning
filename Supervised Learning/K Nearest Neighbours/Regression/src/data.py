"""Data loading utilities for the diabetes regression dataset."""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from .config import CONFIG, KNNRegressionConfig


def _load_from_source(config: KNNRegressionConfig) -> pd.DataFrame:
    dataset = load_diabetes(as_frame=True)
    df = dataset.frame.copy()
    df = df.rename(columns={"target": config.target_column})
    config.data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.data_path, index=False)
    return df


def load_dataset(config: KNNRegressionConfig = CONFIG) -> pd.DataFrame:
    """Load the dataset from disk, fetching from scikit-learn if needed."""
    if config.data_path.exists():
        df = pd.read_csv(config.data_path)
    else:
        df = _load_from_source(config)
    return df


def build_features(
    df: pd.DataFrame, config: KNNRegressionConfig = CONFIG
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into feature matrix and numeric target."""
    X = df[list(config.feature_columns)]
    y = df[config.target_column]
    return X, y


def train_validation_split(
    config: KNNRegressionConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return train/validation split with a fixed random seed."""
    df = load_dataset(config)
    X, y = build_features(df, config)
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )
