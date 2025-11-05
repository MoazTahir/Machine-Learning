"""Command-line entry point for training the linear regression model."""
from __future__ import annotations

import json

from .pipeline import train_and_persist


def main() -> dict[str, float]:
    """Train the pipeline and return evaluation metrics."""
    metrics = train_and_persist()
    return metrics


if __name__ == "__main__":
    results = main()
    print(json.dumps(results, indent=2))
