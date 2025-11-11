"""Configuration for Platt scaling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PlattScalingConfig:
    target_column: str = "label"
    feature_columns: tuple[str, ...] = ("feature_1", "feature_2")
    test_size: float = 0.3
    random_state: int = 42
    base_estimator_C: float = 1.0
    data_path: Path = ALGORITHM_ROOT / "data" / "platt_scaling.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "platt_scaling_model.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_paths(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = PlattScalingConfig()
CONFIG.ensure_paths()
