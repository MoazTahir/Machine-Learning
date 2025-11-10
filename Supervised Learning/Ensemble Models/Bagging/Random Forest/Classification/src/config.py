"""Configuration for the breast cancer random forest classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RandomForestClassificationConfig:
    """Immutable configuration shared across the random forest classifier."""

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
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | float = "sqrt"
    bootstrap: bool = True
    class_weight: str | None = "balanced_subsample"
    data_path: Path = ALGORITHM_ROOT / "data" / "breast_cancer.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "random_forest_classifier.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = RandomForestClassificationConfig()
CONFIG.ensure_directories()
