"""Configuration for the diabetes K-Nearest Neighbours regressor."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class KNNRegressionConfig:
    """Immutable configuration shared across the KNN regression pipeline."""

    target_column: str = "disease_progression"
    feature_columns: tuple[str, ...] = (
        "age",
        "sex",
        "bmi",
        "bp",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
    )
    test_size: float = 0.2
    random_state: int = 42
    n_neighbors: int = 9
    weights: str = "distance"
    metric: str = "minkowski"
    data_path: Path = ALGORITHM_ROOT / "data" / "diabetes.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "knn_regressor.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = KNNRegressionConfig()
CONFIG.ensure_directories()
