"""Quick demonstration of the wine KNN classifier pipeline."""
from __future__ import annotations

import json

from src.data import load_dataset
from src.pipeline import WineKNNPipeline


def main() -> None:
    pipeline = WineKNNPipeline()
    metrics = pipeline.train()

    df = load_dataset(pipeline.config)
    sample = df.sample(1, random_state=pipeline.config.random_state)
    features = sample[pipeline.config.feature_columns]

    predicted_label = pipeline.pipeline.predict(features)[0]  # type: ignore[arg-type]
    probabilities = pipeline.pipeline.predict_proba(features)[0]  # type: ignore[arg-type]

    print(
        json.dumps(
            {
                "sample": features.iloc[0].to_dict(),
                "predicted_label": predicted_label,
                "class_probabilities": {
                    str(label): float(prob)
                    for label, prob in zip(pipeline.pipeline.classes_, probabilities)  # type: ignore[union-attr]
                },
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
