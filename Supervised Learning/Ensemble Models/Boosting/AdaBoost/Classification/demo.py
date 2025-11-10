"""Quick demo for the breast cancer AdaBoost classifier."""
from __future__ import annotations

from src.inference import AdaBoostClassificationRequest, get_service


def main() -> None:
    service = get_service()
    sample = AdaBoostClassificationRequest(
        mean_radius=17.9,
        mean_texture=10.1,
        mean_perimeter=123.4,
        mean_area=1004.0,
        mean_smoothness=0.118,
        mean_compactness=0.27,
        mean_concavity=0.30,
        mean_concave_points=0.15,
        mean_symmetry=0.24,
        mean_fractal_dimension=0.079,
        radius_error=1.10,
        texture_error=0.90,
        perimeter_error=8.60,
        area_error=153.0,
        smoothness_error=0.0064,
        compactness_error=0.049,
        concavity_error=0.054,
        concave_points_error=0.016,
        symmetry_error=0.030,
        fractal_dimension_error=0.0062,
        worst_radius=25.4,
        worst_texture=17.3,
        worst_perimeter=185.0,
        worst_area=2020.0,
        worst_smoothness=0.162,
        worst_compactness=0.666,
        worst_concavity=0.712,
        worst_concave_points=0.265,
        worst_symmetry=0.460,
        worst_fractal_dimension=0.119,
    )
    response = service.predict(sample)
    print(response.model_dump())


if __name__ == "__main__":
    main()
