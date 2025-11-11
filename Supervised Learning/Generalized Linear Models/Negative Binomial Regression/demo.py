"""Demonstration script for negative binomial regression."""
from __future__ import annotations

import json

import pandas as pd

from src.pipeline import NegativeBinomialPipeline


def main() -> None:
    pipeline = NegativeBinomialPipeline()
    metrics = pipeline.train()

    sample_df = pd.DataFrame(
        [[5, 8.0, 2]],
        columns=["weekday", "exposure_hours", "promotions"],
    )
    design_matrix, _ = pipeline.build_design_matrices(sample_df, [0])
    prediction = float(pipeline.result.predict(design_matrix)[0])  # type: ignore[arg-type]

    print(
        json.dumps(
            {
                "input": sample_df.iloc[0].to_dict(),
                "expected_count": prediction,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
