"""Quick demonstration of the diabetes KNN regressor pipeline."""
from __future__ import annotations

import json

from src.data import load_dataset
from src.pipeline import DiabetesKNNPipeline


def main() -> None:
    pipeline = DiabetesKNNPipeline()
    metrics = pipeline.train()

    df = load_dataset(pipeline.config)
    sample = df.sample(1, random_state=pipeline.config.random_state)
    features = sample[pipeline.config.feature_columns]

    predicted_value = float(pipeline.pipeline.predict(features)[0])  # type: ignore[arg-type]

    print(
        json.dumps(
            {
                "sample": features.iloc[0].to_dict(),
                "predicted_value": predicted_value,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
