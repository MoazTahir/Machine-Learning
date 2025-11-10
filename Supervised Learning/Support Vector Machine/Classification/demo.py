"""Quick demonstration of the breast cancer SVM pipeline."""
from __future__ import annotations

import json

import pandas as pd

from src.pipeline import BreastCancerSVMPipeline


def main() -> None:
    pipeline = BreastCancerSVMPipeline()
    metrics = pipeline.train()

    sample = pd.DataFrame(
        [
            {
                "mean_radius": 20.57,
                "mean_texture": 17.77,
                "mean_perimeter": 132.90,
                "mean_area": 1326.0,
                "mean_smoothness": 0.08474,
                "mean_compactness": 0.07864,
                "mean_concavity": 0.0869,
                "mean_concave_points": 0.07017,
                "mean_symmetry": 0.1812,
                "mean_fractal_dimension": 0.05667,
                "radius_error": 1.095,
                "texture_error": 0.9053,
                "perimeter_error": 8.589,
                "area_error": 153.4,
                "smoothness_error": 0.006399,
                "compactness_error": 0.04904,
                "concavity_error": 0.05373,
                "concave_points_error": 0.01587,
                "symmetry_error": 0.03003,
                "fractal_dimension_error": 0.006193,
                "worst_radius": 25.38,
                "worst_texture": 17.33,
                "worst_perimeter": 184.6,
                "worst_area": 2019.0,
                "worst_smoothness": 0.1622,
                "worst_compactness": 0.6656,
                "worst_concavity": 0.7119,
                "worst_concave_points": 0.2654,
                "worst_symmetry": 0.4601,
                "worst_fractal_dimension": 0.1189,
            }
        ]
    )

    proba_malignant = float(pipeline.pipeline.predict_proba(sample)[0][1])  # type: ignore[arg-type]
    label = "malignant" if proba_malignant >= 0.5 else "benign"

    print(
        json.dumps(
            {
                "sample": sample.iloc[0].to_dict(),
                "predicted_label": label,
                "probability_malignant": proba_malignant,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
