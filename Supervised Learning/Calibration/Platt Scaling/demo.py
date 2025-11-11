"""Demonstration script for Platt scaling."""
from __future__ import annotations

import json

from src.pipeline import PlattScalingPipeline


def main() -> None:
    pipeline = PlattScalingPipeline()
    metrics = pipeline.train()

    sample = [[1.6, 3.6]]
    probability = float(pipeline.pipeline.predict_proba(sample)[0, 1])  # type: ignore[arg-type]

    print(
        json.dumps(
            {
                "input": {"feature_1": sample[0][0], "feature_2": sample[0][1]},
                "calibrated_probability": probability,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
