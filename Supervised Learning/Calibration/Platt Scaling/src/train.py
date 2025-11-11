"""CLI entry point for Platt scaling calibration."""
from __future__ import annotations

import json

from .pipeline import train_and_persist


def main() -> dict[str, float]:
    return train_and_persist()


if __name__ == "__main__":
    results = main()
    print(json.dumps(results, indent=2))
