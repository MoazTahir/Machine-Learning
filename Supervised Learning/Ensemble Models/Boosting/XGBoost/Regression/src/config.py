"""Configuration for the California housing XGBoost regressor."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class XGBoostRegressionConfig:
    """Immutable configuration for the XGBoost regressor."""

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
    learning_rate: float = 0.08
    n_estimators: int = 600
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    data_path: Path = ALGORITHM_ROOT / "data" / "california_housing.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "xgboost_regressor.json"
    metrics_path: Path = artifact_dir / "metrics.json"
    feature_importances_path: Path = artifact_dir / "feature_importances.json"
    learning_curve_path: Path = artifact_dir / "learning_curve.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = XGBoostRegressionConfig()
CONFIG.ensure_directories()
