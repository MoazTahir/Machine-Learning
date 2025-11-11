"""Configuration for the ridge regression module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RidgeRegressionConfig:
    """Static configuration shared across the ridge regression pipeline."""

    target_column: str = "salary"
    feature_columns: tuple[str, ...] = (
        "experience",
        "feature_multicollinear",
        "feature_noise",
    )
    test_size: float = 0.2
    random_state: int = 42
    alpha: float = 10.0
    data_path: Path = ALGORITHM_ROOT / "data" / "ridge_regression.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "ridge_regression_model.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_paths(self) -> None:
        """Create required directories before writing artifacts."""

        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = RidgeRegressionConfig()
CONFIG.ensure_paths()
