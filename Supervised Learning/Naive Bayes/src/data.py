"""Data loading utilities for the mushroom dataset."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CONFIG, NaiveBayesConfig


def load_dataset(config: NaiveBayesConfig = CONFIG) -> pd.DataFrame:
    """Load the mushroom dataset and normalise missing markers."""
    df = pd.read_csv(config.data_path)
    return df.replace("?", pd.NA)


def build_features(
    df: pd.DataFrame, config: NaiveBayesConfig = CONFIG
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into feature matrix and binary target."""
    X = df[list(config.feature_columns)]
    y = (df[config.target_column] == "p").astype(int)
    return X, y


def train_validation_split(
    config: NaiveBayesConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return train/validation split using stratified sampling."""
    df = load_dataset(config)
    X, y = build_features(df, config)
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
