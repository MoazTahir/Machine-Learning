"""Data loading utilities for the breast cancer dataset."""
from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from .config import CONFIG, SVMConfig


def _normalise_column(name: str) -> str:
    cleaned = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("/", "_")
    )
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _load_from_source(config: SVMConfig) -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    rename_fn: Callable[[str], str] = _normalise_column
    df = df.rename(columns={col: rename_fn(col) for col in df.columns})
    df = df.rename(columns={"target": config.target_column})
    df[config.target_column] = df[config.target_column].map({0: "malignant", 1: "benign"})
    config.data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.data_path, index=False)
    return df


def load_dataset(config: SVMConfig = CONFIG) -> pd.DataFrame:
    """Load the dataset from disk, downloading from scikit-learn if needed."""
    if config.data_path.exists():
        df = pd.read_csv(config.data_path)
    else:
        df = _load_from_source(config)
    return df


def build_features(
    df: pd.DataFrame, config: SVMConfig = CONFIG
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into feature matrix and binary target."""
    X = df[list(config.feature_columns)]
    y = (df[config.target_column] == config.positive_label).astype(int)
    return X, y


def train_validation_split(
    config: SVMConfig = CONFIG,
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
