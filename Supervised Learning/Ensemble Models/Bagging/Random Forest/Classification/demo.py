"""Quick demonstration script for the breast cancer random forest classifier."""
from __future__ import annotations

from src.inference import RandomForestClassificationRequest, get_service


def main() -> None:
    service = get_service()
    sample = RandomForestClassificationRequest(
        mean_radius=14.05,
        mean_texture=14.96,
        mean_perimeter=92.25,
        mean_area=606.5,
        mean_smoothness=0.09711,
        mean_compactness=0.06154,
        mean_concavity=0.01981,
        mean_concave_points=0.01768,
        mean_symmetry=0.1837,
        mean_fractal_dimension=0.05936,
        radius_error=0.5437,
        texture_error=0.7339,
        perimeter_error=3.398,
        area_error=51.32,
        smoothness_error=0.005225,
        compactness_error=0.01855,
        concavity_error=0.01988,
        concave_points_error=0.00653,
        symmetry_error=0.02045,
        fractal_dimension_error=0.003582,
        worst_radius=15.3,
        worst_texture=19.35,
        worst_perimeter=102.5,
        worst_area=729.8,
        worst_smoothness=0.1347,
        worst_compactness=0.114,
        worst_concavity=0.07081,
        worst_concave_points=0.03354,
        worst_symmetry=0.230,
        worst_fractal_dimension=0.07699,
    )
    response = service.predict(sample)
    print(response.model_dump())


if __name__ == "__main__":
    main()
