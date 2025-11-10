"""Configuration for the California housing AdaBoost regressor."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AdaBoostRegressionConfig:
    """Immutable configuration for the AdaBoost regressor."""

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
    n_estimators: int = 400
    learning_rate: float = 0.5
    loss: str = "square"
    estimator_max_depth: int = 4
    data_path: Path = ALGORITHM_ROOT / "data" / "california_housing.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "adaboost_regressor.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"
    staged_metrics_path: Path = artifact_dir / "learning_curve.json"
    feature_importances_path: Path = artifact_dir / "feature_importances.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = AdaBoostRegressionConfig()
CONFIG.ensure_directories()
