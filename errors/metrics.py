from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

ArrayLike = Iterable[float] | np.ndarray


def _to_numpy(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape == ():  # guard against scalar inputs
        raise ValueError("Expected an array-like with at least one element.")
    return array


def _validate_pairs(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    if y_t.shape != y_p.shape:
        raise ValueError("y_true and y_pred must share the same shape.")
    return y_t, y_p


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_t, y_p = _validate_pairs(y_true, y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_t, y_p = _validate_pairs(y_true, y_pred)
    return float(np.mean(np.square(y_t - y_p)))


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_percentage_error(y_true: ArrayLike, y_pred: ArrayLike, *, epsilon: float = 1e-12) -> float:
    y_t, y_p = _validate_pairs(y_true, y_pred)
    if np.any(np.equal(np.abs(y_t), 0.0)):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive when y_true contains zeros.")
    denominator = np.where(np.abs(y_t) < epsilon, epsilon, np.abs(y_t))
    return float(np.mean(np.abs((y_t - y_p) / denominator)) * 100)


def symmetric_mean_absolute_percentage_error(y_true: ArrayLike, y_pred: ArrayLike, *, epsilon: float = 1e-12) -> float:
    y_t, y_p = _validate_pairs(y_true, y_pred)
    denominator = (np.abs(y_t) + np.abs(y_p))
    denominator = np.where(denominator < epsilon, epsilon, denominator)
    return float(np.mean((np.abs(y_t - y_p) * 200) / denominator))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_t, y_p = _validate_pairs(y_true, y_pred)
    ss_res = np.sum(np.square(y_t - y_p))
    ss_tot = np.sum(np.square(y_t - np.mean(y_t)))
    if ss_tot == 0:
        raise ValueError("R^2 is undefined when y_true has zero variance.")
    return float(1 - ss_res / ss_tot)


def huber_loss(y_true: ArrayLike, y_pred: ArrayLike, *, delta: float = 1.0) -> float:
    if delta <= 0:
        raise ValueError("delta must be positive.")
    y_t, y_p = _validate_pairs(y_true, y_pred)
    residual = y_t - y_p
    mask = np.abs(residual) <= delta
    squared = 0.5 * np.square(residual)
    linear = delta * (np.abs(residual) - 0.5 * delta)
    return float(np.mean(np.where(mask, squared, linear)))


def log_cosh_loss(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_t, y_p = _validate_pairs(y_true, y_pred)
    residual = y_t - y_p
    return float(np.mean(np.log(np.cosh(residual))))


def quantile_loss(y_true: ArrayLike, y_pred: ArrayLike, *, quantile: float) -> float:
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie in the open interval (0, 1).")
    y_t, y_p = _validate_pairs(y_true, y_pred)
    residual = y_t - y_p
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1) * residual)))


def mean_absolute_scaled_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    seasonality: int = 1,
) -> float:
    if seasonality <= 0:
        raise ValueError("seasonality must be positive.")
    y_t, y_p = _validate_pairs(y_true, y_pred)
    if y_t.shape[0] <= seasonality:
        raise ValueError("y_true must contain more points than the seasonality window.")
    numerator = np.mean(np.abs(y_t - y_p))
    denominator = np.mean(np.abs(y_t[seasonality:] - y_t[:-seasonality]))
    if denominator == 0:
        raise ValueError("MASE denominator is zero; the naive seasonal forecast fits perfectly.")
    return float(numerator / denominator)


def binary_cross_entropy(y_true: ArrayLike, y_prob: ArrayLike, *, epsilon: float = 1e-15) -> float:
    y_t, y_p = _validate_pairs(y_true, y_prob)
    if np.any((y_t < 0) | (y_t > 1)):
        raise ValueError("y_true must contain binary targets in [0, 1].")
    clipped = np.clip(y_p, epsilon, 1 - epsilon)
    loss = -(y_t * np.log(clipped) + (1 - y_t) * np.log(1 - clipped))
    return float(np.mean(loss))


def categorical_cross_entropy(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    epsilon: float = 1e-15,
) -> float:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)
    if y_t.shape != y_p.shape:
        raise ValueError("y_true and y_prob must share the same shape.")
    if y_t.ndim != 2:
        raise ValueError("Expected two-dimensional arrays for categorical targets.")
    clipped = np.clip(y_p, epsilon, 1 - epsilon)
    if not np.allclose(np.sum(y_t, axis=1), 1.0):
        raise ValueError("y_true rows must sum to 1 (one-hot or probability targets).")
    loss = -np.sum(y_t * np.log(clipped), axis=1)
    return float(np.mean(loss))


def hinge_loss(y_true: ArrayLike, decision: ArrayLike) -> float:
    y_t, y_f = _validate_pairs(y_true, decision)
    if np.any(~np.isin(y_t, (-1, 1))):
        raise ValueError("y_true must contain only -1 and 1 labels for hinge loss.")
    return float(np.mean(np.maximum(0.0, 1.0 - y_t * y_f)))


METRIC_REGISTRY: Mapping[str, callable] = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error,
    "mape": mean_absolute_percentage_error,
    "smape": symmetric_mean_absolute_percentage_error,
    "r2": r2_score,
    "huber": huber_loss,
    "log_cosh": log_cosh_loss,
    "quantile": quantile_loss,
    "mase": mean_absolute_scaled_error,
    "binary_cross_entropy": binary_cross_entropy,
    "categorical_cross_entropy": categorical_cross_entropy,
    "hinge": hinge_loss,
}

__all__ = [
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_percentage_error",
    "symmetric_mean_absolute_percentage_error",
    "r2_score",
    "huber_loss",
    "log_cosh_loss",
    "quantile_loss",
    "mean_absolute_scaled_error",
    "binary_cross_entropy",
    "categorical_cross_entropy",
    "hinge_loss",
    "METRIC_REGISTRY",
]
