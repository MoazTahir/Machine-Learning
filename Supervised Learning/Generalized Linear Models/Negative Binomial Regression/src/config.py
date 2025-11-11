"""Configuration for negative binomial regression."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class NegativeBinomialConfig:
    target_column: str = "count_events"
    feature_columns: tuple[str, ...] = (
        "weekday",
        "exposure_hours",
        "promotions",
    )
    test_size: float = 0.2
    random_state: int = 42
    data_path: Path = ALGORITHM_ROOT / "data" / "negative_binomial_counts.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "negative_binomial_model.pkl"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_paths(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = NegativeBinomialConfig()
CONFIG.ensure_paths()
