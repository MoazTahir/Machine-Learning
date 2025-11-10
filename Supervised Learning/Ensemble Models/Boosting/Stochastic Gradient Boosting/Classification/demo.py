"""Quick demo for the wine stochastic gradient boosting classifier."""
from __future__ import annotations

from src.inference import (
    StochasticGradientBoostingClassificationRequest,
    get_service,
)


def main() -> None:
    service = get_service()
    sample = StochasticGradientBoostingClassificationRequest(
        alcohol=13.2,
        malic_acid=1.78,
        ash=2.14,
        alcalinity_of_ash=11.2,
        magnesium=100.0,
        total_phenols=2.65,
        flavanoids=2.76,
        nonflavanoid_phenols=0.26,
        proanthocyanins=1.28,
        color_intensity=4.38,
        hue=1.05,
        od280_od315_of_diluted_wines=3.4,
        proline=1050.0,
    )
    response = service.predict(sample)
    print(response.model_dump())


if __name__ == "__main__":
    main()
