"""Configuration for the logistic regression heart disease classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Static configuration shared across the logistic regression pipeline."""

    target_column: str = "target"
    feature_columns: tuple[str, ...] = (
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    )
    test_size: float = 0.2
    random_state: int = 42
    data_path: Path = ALGORITHM_ROOT / "data" / "heart.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "logistic_regression_model.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = LogisticRegressionConfig()
CONFIG.ensure_directories()
