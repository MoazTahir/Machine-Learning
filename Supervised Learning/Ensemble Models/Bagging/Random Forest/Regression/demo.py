"""Quick demonstration script for the California housing random forest regressor."""
from __future__ import annotations

from src.inference import RandomForestRegressionRequest, get_service


def main() -> None:
    service = get_service()
    sample = RandomForestRegressionRequest(
        median_income=6.32,
        house_age=34.0,
        average_rooms=5.4,
        average_bedrooms=1.1,
        population=980.0,
        average_occupancy=2.8,
        latitude=37.88,
        longitude=-122.23,
    )
    response = service.predict(sample)
    print(response.model_dump())


if __name__ == "__main__":
    main()
