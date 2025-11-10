"""Configuration for the wine stochastic gradient boosting classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class StochasticGradientBoostingClassificationConfig:
    """Immutable configuration for the stochastic gradient boosting classifier."""

    target_column: str = "class_label"
    class_names: tuple[str, ...] = ("class_0", "class_1", "class_2")
    feature_columns: tuple[str, ...] = (
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280_od315_of_diluted_wines",
        "proline",
    )
    test_size: float = 0.2
    random_state: int = 42
    learning_rate: float = 0.05
    n_estimators: int = 400
    max_depth: int = 3
    subsample: float = 0.6
    max_features: str | None = "sqrt"
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    data_path: Path = ALGORITHM_ROOT / "data" / "wine.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "stochastic_gradient_boosting_classifier.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = StochasticGradientBoostingClassificationConfig()
CONFIG.ensure_directories()
