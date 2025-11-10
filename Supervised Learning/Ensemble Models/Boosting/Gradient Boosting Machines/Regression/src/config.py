"""Configuration for the California housing gradient boosting regressor."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class GradientBoostingRegressionConfig:
    """Immutable configuration shared across the gradient boosting regressor."""

    target_column: str = "median_house_value"
    feature_columns: tuple[str, ...] = (
        "median_income",
        "house_age",
        "average_rooms",
        "average_bedrooms",
        "population",
        "average_occupancy",
        "latitude",
        "longitude",
    )
    test_size: float = 0.2
    random_state: int = 42
    learning_rate: float = 0.05
    n_estimators: int = 600
    max_depth: int = 3
    subsample: float = 0.9
    max_features: str | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    loss: str = "squared_error"
    data_path: Path = ALGORITHM_ROOT / "data" / "california_housing.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "gradient_boosting_regressor.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = GradientBoostingRegressionConfig()
CONFIG.ensure_directories()
