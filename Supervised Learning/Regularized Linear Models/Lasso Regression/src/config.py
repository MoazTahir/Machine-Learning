"""Configuration for the lasso regression module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class LassoRegressionConfig:
    """Static configuration shared across the lasso pipeline."""

    target_column: str = "salary"
    feature_columns: tuple[str, ...] = (
        "experience",
        "feature_sparse1",
        "feature_sparse2",
    )
    test_size: float = 0.2
    random_state: int = 42
    alpha: float = 0.1
    max_iter: int = 10000
    data_path: Path = ALGORITHM_ROOT / "data" / "lasso_regression.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "lasso_regression_model.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_paths(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = LassoRegressionConfig()
CONFIG.ensure_paths()
