"""Data utilities for agglomerative clustering."""
from __future__ import annotations

import pandas as pd

from .config import CONFIG, AgglomerativeConfig


def load_dataset(config: AgglomerativeConfig = CONFIG) -> pd.DataFrame:
    return pd.read_csv(config.data_path)
