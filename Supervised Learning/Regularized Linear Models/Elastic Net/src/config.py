"""Configuration for the elastic net module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ElasticNetConfig:
    target_column: str = "salary"
    feature_columns: tuple[str, ...] = (
        "experience",
        "feature_group1",
        "feature_group2",
        "feature_noise",
    )
    test_size: float = 0.2
    random_state: int = 42
    alpha: float = 0.5
    l1_ratio: float = 0.5
    max_iter: int = 10000
    data_path: Path = ALGORITHM_ROOT / "data" / "elastic_net_regression.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "elastic_net_model.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_paths(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = ElasticNetConfig()
CONFIG.ensure_paths()
