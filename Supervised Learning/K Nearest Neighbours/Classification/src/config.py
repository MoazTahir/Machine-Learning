"""Configuration for the wine K-Nearest Neighbours classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class KNNClassificationConfig:
    """Immutable configuration shared across the KNN classification pipeline."""

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
    n_neighbors: int = 7
    weights: str = "distance"
    metric: str = "minkowski"
    data_path: Path = ALGORITHM_ROOT / "data" / "wine.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "knn_classifier.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = KNNClassificationConfig()
CONFIG.ensure_directories()
