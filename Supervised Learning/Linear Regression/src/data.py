"""Data loading utilities for the linear regression project."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CONFIG, LinearRegressionConfig


def load_dataset(config: LinearRegressionConfig = CONFIG) -> pd.DataFrame:
    """Load the salary dataset from disk."""
    return pd.read_csv(config.data_path)


def build_features(
    df: pd.DataFrame, config: LinearRegressionConfig = CONFIG
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into feature matrix and target vector."""
    X = df[list(config.feature_columns)]
    y = df[config.target_column]
    return X, y


def train_validation_split(
    config: LinearRegressionConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return train/validation split using the configured proportions."""
    df = load_dataset(config)
    X, y = build_features(df, config)
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )
