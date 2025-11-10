<!-- markdownlint-disable MD013 -->
# Prophet Forecasting — AirPassengers

This module packages Facebook Prophet (now simply Prophet) for the AirPassengers dataset. It mirrors the repository’s production-ready layout: configuration, scripted training, persisted artefacts, FastAPI integration, notebook exploration, and a CLI demo. Use it to experiment with additive or multiplicative seasonality, changepoint control, and probabilistic forecast intervals.

---

## Learning Objectives

- Fit a Prophet model with multiplicative yearly seasonality to airline passenger data.
- Evaluate forecasts against a held-out slice and persist metrics.
- Return median forecasts plus lower/upper bounds through the shared FastAPI registry.
- Extend the baseline with custom holidays, extra regressors, or hyperparameter sweeps.

---

## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and persist the model**

   ```bash
   python "Supervised Learning/Time Series Forcasting/Prophet/src/train.py"
   ```

   Example metrics output:

   ```json
   {
     "mae": 10.8,
     "rmse": 14.9,
     "mape": 4.9
   }
   ```

3. **Launch the FastAPI service**

   ```bash
   python -m fastapi_app.main
   ```

4. **Request a forecast**

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/prophet_forecast" \
        -H "Content-Type: application/json" \
        -d '{"horizon": 12}'
   ```

   Prophet returns median forecasts (`yhat`) and uncertainty intervals (`yhat_lower`, `yhat_upper`).

5. **Open the notebook**

   `notebooks/prophet_forecasting.ipynb` mirrors the scripted pipeline, including trend/seasonality plots and residual checks.

---

## Mathematical Foundations

Prophet decomposes a time series into trend, seasonality, and holiday components:

$$
y(t) = g(t) + s(t) + h(t) + \varepsilon_t.
$$

- $g(t)$ models the trend via piecewise linear or logistic growth with changepoints.
- $s(t)$ captures seasonal patterns using Fourier series.
- $h(t)$ encodes user-defined holidays or events.

The model is fit using Stan, providing posterior uncertainty estimates that translate into forecast intervals.

### Plain-Language Intuition

Prophet decomposes the series into building blocks you can reason about: a smooth trend curve that changes slope at a handful of data-driven changepoints, repeating seasonal cycles drawn with Fourier curves, and optional holiday pulses. Forecasting is as simple as extending those components into the future and summing them, which keeps the predictions interpretable (“trend explains 70%, seasonality swings ±15%, holidays add the rest”).

### When to Reach for Prophet

- You need production forecasts quickly with minimal tuning and solid defaults.
- The business wants interpretable components (trend vs. seasonality vs. events) and uncertainty intervals out of the box.
- The series has irregular sampling, missing values, or occasional outliers that would break stricter classical models.

### Strengths & Limitations

- **Strengths:** Fast to prototype, robust to gaps and outliers, supports multiple seasonalities and holiday effects, automatically produces prediction intervals.
- **Limitations:** Assumes piecewise-linear or logistic growth; struggles with sudden regime changes unless changepoint priors are relaxed; less precise for very short histories compared to ARIMA family.

### Practical Tuning Tips

- Increase `changepoint_prior_scale` for more flexible trend changes; decrease it to smooth the curve.
- Switch `seasonality_mode` between `additive` and `multiplicative` depending on whether seasonal amplitude grows with the level.
- Use `add_country_holidays` or custom event tables to capture domain-specific peaks (e.g., summer travel, holidays).

### Diagnostics & Monitoring

- Plot Prophet’s built-in component charts (`model.plot_components`) to confirm the learned trend and seasonal shapes align with intuition.
- Review residual histograms and coverage of `yhat_lower`/`yhat_upper` to ensure uncertainty intervals are calibrated.
- Run Prophet’s `cross_validation` helper for rolling-origin evaluation; track MAPE or RMSE over time to catch drift in production.

---

## Dataset

- **Source**: AirPassengers (monthly airline passengers, 1949–1960).
- **Cache**: `data/air_passengers.csv` (`ds`, `y` columns for Prophet).
- **Frequency**: Monthly (`MS`).

---

## Repository Layout

```
Prophet/
├── README.md
├── demo.py
├── data/
│   └── air_passengers.csv
├── notebooks/
│   └── prophet_forecasting.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── pipeline.py
│   └── train.py
└── artifacts/
    └── .gitkeep
```

---

## Implementation Walkthrough

- `config.py` — centralises Prophet hyperparameters (seasonality mode, changepoint prior scale), file paths, and default forecast horizon.
- `data.py` — loads the AirPassengers CSV, renames columns to Prophet’s expected `ds`/`y`, and performs deterministic train/test splits.
- `pipeline.py` — constructs and fits the Prophet model, logs diagnostics, exports artefacts via joblib and JSON, and exposes a reusable `forecast` helper.
- `inference.py` — wraps the trained model in a FastAPI-friendly service with Pydantic schemas that surface `yhat`, `yhat_lower`, and `yhat_upper` bands.
- `train.py` — CLI entry point that orchestrates training and metric persistence for reproducible runs.
- `demo.py` — offers a quick smoke test that trains (if necessary) and prints a 12-month projection.

This mirrors the supervised-learning pattern, so swapping models in the FastAPI registry or experimentation workflow stays frictionless.

---

## Extensions

1. **Holiday calendars** — inject known travel peaks to improve accuracy.
2. **Scenario analysis** — adjust growth rate priors or cap/floor for capacity planning.
3. **Cross-validation** — use Prophet’s built-in `cross_validation` utilities for rolling evaluation.

---

## References

- Sean J. Taylor, Benjamin Letham. "Forecasting at scale." *The American Statistician*, 2018.
- Prophet documentation: [https://facebook.github.io/prophet/](https://facebook.github.io/prophet/)
- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*.
