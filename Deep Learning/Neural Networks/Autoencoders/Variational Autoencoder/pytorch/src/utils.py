"""Utilities for the PyTorch variational autoencoder."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def psnr(recon: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    mse = torch.mean((recon - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(data_range) - 10 * math.log10(mse)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
