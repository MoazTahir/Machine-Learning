"""Quick demonstration script for ridge regression."""
from __future__ import annotations

import json

from src.pipeline import RidgeRegressionPipeline


def main() -> None:
    pipeline = RidgeRegressionPipeline()
    metrics = pipeline.train()

    sample = [[6.5, 6.2, 0.05]]
    prediction = float(pipeline.pipeline.predict(sample)[0])  # type: ignore[arg-type]

    print(
        json.dumps(
            {
                "input": {
                    "experience": sample[0][0],
                    "feature_multicollinear": sample[0][1],
                    "feature_noise": sample[0][2],
                },
                "prediction": prediction,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()