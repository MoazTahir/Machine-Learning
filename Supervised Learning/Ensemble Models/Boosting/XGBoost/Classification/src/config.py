"""Configuration for the wine XGBoost classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class XGBoostClassificationConfig:
    """Immutable configuration for the XGBoost classifier."""

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
    n_estimators: int = 350
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    data_path: Path = ALGORITHM_ROOT / "data" / "wine.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "xgboost_classifier.json"
    metrics_path: Path = artifact_dir / "metrics.json"
    feature_importances_path: Path = artifact_dir / "feature_importances.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = XGBoostClassificationConfig()
CONFIG.ensure_directories()
