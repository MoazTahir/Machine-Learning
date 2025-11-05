"""Configuration for the mushroom edibility Naive Bayes classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class NaiveBayesConfig:
    """Immutable configuration shared across the Naive Bayes pipeline."""

    target_column: str = "class"
    feature_columns: tuple[str, ...] = (
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat",
    )
    test_size: float = 0.2
    random_state: int = 42
    data_path: Path = ALGORITHM_ROOT / "data" / "mushrooms.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "naive_bayes_model.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = NaiveBayesConfig()
CONFIG.ensure_directories()
