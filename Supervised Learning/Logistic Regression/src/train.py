"""Command-line entry point for training the logistic regression model."""
from __future__ import annotations

import json

from .pipeline import train_and_persist


def main() -> dict[str, float]:
    """Train the pipeline and return evaluation metrics."""
    return train_and_persist()


if __name__ == "__main__":
    metrics = main()
    print(json.dumps(metrics, indent=2))
