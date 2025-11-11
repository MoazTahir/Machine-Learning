"""Demonstration script for elastic net regression."""
from __future__ import annotations

import json

from src.pipeline import ElasticNetPipeline


def main() -> None:
    pipeline = ElasticNetPipeline()
    metrics = pipeline.train()

    sample = [[6.5, 6.2, 6.1, 0.03]]
    prediction = float(pipeline.pipeline.predict(sample)[0])  # type: ignore[arg-type]

    print(
        json.dumps(
            {
                "input": {
                    "experience": sample[0][0],
                    "feature_group1": sample[0][1],
                    "feature_group2": sample[0][2],
                    "feature_noise": sample[0][3],
                },
                "prediction": prediction,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
