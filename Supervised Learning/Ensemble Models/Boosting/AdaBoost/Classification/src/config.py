"""Configuration for the breast cancer AdaBoost classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AdaBoostClassificationConfig:
    """Immutable configuration for the AdaBoost classifier."""

    target_column: str = "diagnosis"
    positive_label: str = "malignant"
    class_names: tuple[str, ...] = ("malignant", "benign")
    feature_columns: tuple[str, ...] = (
        "mean_radius",
        "mean_texture",
        "mean_perimeter",
        "mean_area",
        "mean_smoothness",
        "mean_compactness",
        "mean_concavity",
        "mean_concave_points",
        "mean_symmetry",
        "mean_fractal_dimension",
        "radius_error",
        "texture_error",
        "perimeter_error",
        "area_error",
        "smoothness_error",
        "compactness_error",
        "concavity_error",
        "concave_points_error",
        "symmetry_error",
        "fractal_dimension_error",
        "worst_radius",
        "worst_texture",
        "worst_perimeter",
        "worst_area",
        "worst_smoothness",
        "worst_compactness",
        "worst_concavity",
        "worst_concave_points",
        "worst_symmetry",
        "worst_fractal_dimension",
    )
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 250
    learning_rate: float = 0.7
    estimator_max_depth: int = 2
    algorithm: str = "SAMME.R"
    data_path: Path = ALGORITHM_ROOT / "data" / "breast_cancer.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "adaboost_classifier.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = AdaBoostClassificationConfig()
CONFIG.ensure_directories()
