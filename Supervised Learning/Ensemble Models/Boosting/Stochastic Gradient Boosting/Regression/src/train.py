"""CLI entrypoint for the stochastic gradient boosting regressor."""
from __future__ import annotations

import json

from .config import CONFIG
from .pipeline import train_and_persist


def main() -> None:
    metrics = train_and_persist(CONFIG)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
