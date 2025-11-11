"""Data helpers for elastic net regression."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CONFIG, ElasticNetConfig


def load_dataset(config: ElasticNetConfig = CONFIG) -> pd.DataFrame:
    return pd.read_csv(config.data_path)


def build_features(
    df: pd.DataFrame, config: ElasticNetConfig = CONFIG
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[list(config.feature_columns)]
    y = df[config.target_column]
    return X, y


def train_validation_split(
    config: ElasticNetConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_dataset(config)
    X, y = build_features(df, config)
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )
