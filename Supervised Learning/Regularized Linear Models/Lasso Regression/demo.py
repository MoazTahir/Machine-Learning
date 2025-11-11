"""Mini demo for lasso regression."""
from __future__ import annotations

import json

from src.pipeline import LassoRegressionPipeline


def main() -> None:
    pipeline = LassoRegressionPipeline()
    metrics = pipeline.train()

    sample = [[6.5, 0.05, -0.02]]
    prediction = float(pipeline.pipeline.predict(sample)[0])  # type: ignore[arg-type]

    print(
        json.dumps(
            {
                "input": {
                    "experience": sample[0][0],
                    "feature_sparse1": sample[0][1],
                    "feature_sparse2": sample[0][2],
                },
                "prediction": prediction,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
