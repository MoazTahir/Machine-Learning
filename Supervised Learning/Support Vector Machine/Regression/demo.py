"""Quick demonstration of the California housing SVR pipeline."""
from __future__ import annotations

import json

import pandas as pd

from src.pipeline import CaliforniaHousingSVRPipeline


def main() -> None:
    pipeline = CaliforniaHousingSVRPipeline()
    metrics = pipeline.train()

    sample = pd.DataFrame(
        [
            {
                "median_income": 8.3252,
                "house_age": 41.0,
                "average_rooms": 6.9841,
                "average_bedrooms": 1.0238,
                "population": 322.0,
                "average_occupancy": 2.5556,
                "latitude": 37.88,
                "longitude": -122.23,
            }
        ]
    )

    predicted_value = float(pipeline.pipeline.predict(sample)[0])  # type: ignore[arg-type]

    print(
        json.dumps(
            {
                "sample": sample.iloc[0].to_dict(),
                "predicted_value": predicted_value,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
