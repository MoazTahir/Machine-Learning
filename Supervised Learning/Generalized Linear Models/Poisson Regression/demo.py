"""Demonstration script for Poisson regression."""
from __future__ import annotations

import json

from src.pipeline import PoissonRegressionPipeline


def main() -> None:
    pipeline = PoissonRegressionPipeline()
    metrics = pipeline.train()

    sample = [[2, 8.0, 1]]
    expected_count = float(pipeline.pipeline.predict(sample)[0])  # type: ignore[arg-type]

    print(
        json.dumps(
            {
                "input": {
                    "weekday": sample[0][0],
                    "exposure_hours": sample[0][1],
                    "promotions": sample[0][2],
                },
                "expected_count": expected_count,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
