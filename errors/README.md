# Error Metrics Reference

This guide consolidates the most commonly used loss and error metrics across regression and classification workflows. Each entry includes the mathematical definition, a plain-text counterpart for quick reference, typical use cases, and implementation notes that align with the modular utilities in `errors/metrics.py`.

## Getting Started

```python
from errors.metrics import mean_absolute_error, root_mean_squared_error

mae = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
```

- All functions operate on array-like inputs (lists, NumPy arrays, pandas Series).
- Each metric raises `ValueError` when shapes mismatch or required bounds are violated (for example, probabilities outside `[0, 1]`).
- Specify optional parameters such as `delta` for Huber loss or `quantile` for pinball loss to tailor the behaviour.

---

## Regression Metrics

### Mean Absolute Error (MAE)

$$
\operatorname{MAE} = \frac{1}{n} \sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert
$$

```
MAE = (1 / n) * sum_{i=1..n} |y_i - y_hat_i|
```

**When to use:** MAE provides a robust baseline that is less sensitive to outliers than squared-error measures. It is well suited for median-oriented objectives and scenarios where every unit error carries identical cost.

### Mean Squared Error (MSE)

$$
\operatorname{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

```
MSE = (1 / n) * sum_{i=1..n} (y_i - y_hat_i)^2
```

**When to use:** MSE penalises large residuals more aggressively than MAE, making it a natural choice for least-squares regression, linear models, and optimisation procedures that require differentiability.

### Root Mean Squared Error (RMSE)

$$
\operatorname{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

```
RMSE = sqrt( (1 / n) * sum_{i=1..n} (y_i - y_hat_i)^2 )
```

**When to use:** RMSE retains the squared-error emphasis of MSE while restoring the original target units. It is ideal for communicating model accuracy to non-technical stakeholders who need predictions expressed in natural scales.

### Mean Absolute Percentage Error (MAPE)

$$
\operatorname{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left\lvert \frac{y_i - \hat{y}_i}{y_i} \right\rvert
$$

```
MAPE = (100 / n) * sum_{i=1..n} |(y_i - y_hat_i) / y_i|
```

**When to use:** Use MAPE for interpretability in percentage terms, keeping in mind that it is undefined when any `y_i` equals zero and disproportionately penalises small denominators.

### Symmetric Mean Absolute Percentage Error (sMAPE)

$$
\operatorname{sMAPE} = \frac{100}{n} \sum_{i=1}^{n} \frac{\lvert y_i - \hat{y}_i \rvert}{(\lvert y_i \rvert + \lvert \hat{y}_i \rvert)/2}
$$

```
sMAPE = (100 / n) * sum_{i=1..n} ( |y_i - y_hat_i| / ( (|y_i| + |y_hat_i|) / 2 ) )
```

**When to use:** sMAPE mitigates the zero-division issue of MAPE and treats over- and under-forecasts symmetrically, making it a frequent choice in time-series competitions.

### Coefficient of Determination (R²)

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

```
R2 = 1 - ( sum_{i=1..n} (y_i - y_hat_i)^2 ) / ( sum_{i=1..n} (y_i - y_mean)^2 )
```

**When to use:** R² conveys the proportion of variance explained by the model relative to a mean-only baseline. Negative scores indicate that the model underperforms the naive mean predictor.

### Huber Loss

$$
L_{\delta}(r) = \begin{cases}
\frac{1}{2} r^2 & \text{if } \lvert r \rvert \leq \delta, \\
\delta (\lvert r \rvert - \frac{1}{2} \delta) & \text{otherwise}
\end{cases}
$$

```
Huber_delta(r) = 0.5 * r^2                 if |r| <= delta
Huber_delta(r) = delta * (|r| - 0.5 * delta) otherwise
```

**When to use:** Huber loss behaves like MSE near zero residuals and MAE for large errors, offering a robust compromise that preserves differentiability.

### Log-Cosh Loss

$$
L(r) = \sum_{i=1}^{n} \log \cosh(r_i)
$$

```
LogCosh = sum_{i=1..n} log( cosh(r_i) )
```

**When to use:** Log-cosh acts similarly to squared error for small residuals while damping the influence of outliers. It is smooth and well suited for gradient-based optimisation.

### Quantile (Pinball) Loss

$$
L_{\tau}(r) = \sum_{i=1}^{n} \max(\tau r_i, (\tau - 1) r_i)
$$

```
Quantile_tau(r) = sum_{i=1..n} max(tau * r_i, (tau - 1) * r_i)
```

**When to use:** Quantile loss enables quantile regression, estimating conditional quantiles (for example, the 90th percentile). Choose `\tau` in `(0, 1)` to target the desired quantile.

---

## Classification Metrics

### Binary Cross-Entropy (Log Loss)

$$
\operatorname{BCE} = - \frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

```
BCE = -(1 / n) * sum_{i=1..n} [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
```

**When to use:** Binary cross-entropy is the default loss for probabilistic binary classification. It strongly penalises confident yet incorrect predictions and requires probabilities strictly within `(0, 1)`.

### Categorical Cross-Entropy

$$
\operatorname{CCE} = - \frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(p_{ik})
$$

```
CCE = -(1 / n) * sum_{i=1..n} sum_{k=1..K} y_{ik} * log(p_{ik})
```

**When to use:** Apply categorical cross-entropy when working with one-hot encoded targets across multiple classes. It reduces to BCE when `K = 2`.

### Hinge Loss

$$
L = \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i f(x_i))
$$

```
Hinge = (1 / n) * sum_{i=1..n} max(0, 1 - y_i * f(x_i))
```

**When to use:** Hinge loss underpins Support Vector Machines by encouraging large margins. Labels must be encoded as `-1` and `+1`.

---

## Forecasting-Specific Metrics

### Mean Absolute Scaled Error (MASE)

$$
\operatorname{MASE} = \frac{\frac{1}{n} \sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert}{\frac{1}{n - m} \sum_{i=m+1}^{n} \lvert y_i - y_{i-m} \rvert}
$$

```
MASE = ( mean |y_i - y_hat_i| ) / ( mean |y_i - y_{i-m}| )
```

**When to use:** MASE scales absolute error by the in-sample naive seasonal forecast and is handy for comparing models across series with different scales.

---

## Choosing the Right Metric

- **Outlier robustness:** Prefer MAE, Huber, or log-cosh when heavy-tailed noise is present.
- **Probability calibration:** Use cross-entropy variants to reward well-calibrated probability forecasts.
- **Forecasting competitions:** sMAPE and MASE are common in time-series leaderboards due to their scale-independence.
- **Quantile targets:** Quantile loss generalises naturally to prediction intervals and asymmetric risk profiles.

Refer to `errors/metrics.py` for ready-to-use implementations and integration tips.
