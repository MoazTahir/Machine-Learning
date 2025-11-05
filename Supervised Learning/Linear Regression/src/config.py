"""Configuration for the linear regression salary prediction project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class LinearRegressionConfig:
    """Static configuration values used across the linear regression pipeline."""

    target_column: str = "Salary"
    feature_columns: tuple[str, ...] = ("YearsExperience",)
    test_size: float = 0.2
    random_state: int = 42
    data_path: Path = ALGORITHM_ROOT / "data" / "salary_data.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "linear_regression_model.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_paths(self) -> None:
        """Create required directories before writing artifacts."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = LinearRegressionConfig()
CONFIG.ensure_paths()
