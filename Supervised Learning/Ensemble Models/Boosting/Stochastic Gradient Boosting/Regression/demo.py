"""Quick demo for the California housing stochastic gradient boosting regressor."""
from __future__ import annotations

from src.inference import StochasticGradientBoostingRegressionRequest, get_service


def main() -> None:
    service = get_service()
    sample = StochasticGradientBoostingRegressionRequest(
        median_income=5.5,
        house_age=32.0,
        average_rooms=5.9,
        average_bedrooms=1.2,
        population=890.0,
        average_occupancy=2.7,
        latitude=36.8,
        longitude=-121.9,
    )
    response = service.predict(sample)
    print(response.model_dump())


if __name__ == "__main__":
    main()
