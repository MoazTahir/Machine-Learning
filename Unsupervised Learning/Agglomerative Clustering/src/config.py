"""Configuration for agglomerative clustering module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AgglomerativeConfig:
    feature_columns: tuple[str, ...] = ("x", "y")
    data_path: Path = ALGORITHM_ROOT / "data" / "blobs.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "agglomerative_model.joblib"
    metrics_path: Path = artifact_dir / "cluster_report.json"
    n_clusters: int = 3
    linkage: str = "ward"

    def ensure_paths(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


CONFIG = AgglomerativeConfig()
CONFIG.ensure_paths()
