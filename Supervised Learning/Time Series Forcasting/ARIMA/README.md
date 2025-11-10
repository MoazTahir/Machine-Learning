<!-- markdownlint-disable MD013 -->
# ARIMA Forecasting — AirPassengers

This module packages an AutoRegressive Integrated Moving Average (ARIMA) workflow for the classic AirPassengers dataset. It mirrors the supervised-learning layout used throughout the repository: reproducible configuration, scripted training, persisted artefacts, FastAPI integration, an exploratory notebook, and a lightweight demo. Use it to revisit univariate time-series modelling, benchmark seasonal baselines, or deploy forecasts through the shared API layer.

---

## Learning Objectives

- Refresh ARIMA fundamentals: differencing, autoregression, and moving-average components.
- Load and cache monthly airline passenger counts with consistent timestamp indexing.
- Train, evaluate, and persist an `ARIMA(p=1, d=1, q=1)` model using statsmodels.
- Serve rolling forecasts through the FastAPI registry with typed request/response contracts.
- Extend the baseline with alternative orders, diagnostics, or monitoring hooks.

---

## Quickstart

1. **Install shared dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and persist the model**

   ```bash
   python "Supervised Learning/Time Series Forcasting/ARIMA/src/train.py"
   ```

   Example metrics output:

   ```json
   {
     "mae": 16.4,
     "rmse": 20.8,
     "mape": 7.8
   }
   ```

   Artefacts are saved to `artifacts/arima_model.joblib` and `artifacts/metrics.json`. `data/air_passengers.csv` is bundled with the module.

3. **Launch the FastAPI service**

   ```bash
   python -m fastapi_app.main
   ```

4. **Request a forecast** (after registering the ARIMA slug)

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/arima_forecast" \
        -H "Content-Type: application/json" \
        -d '{"horizon": 12}'
   ```

   Sample response:

   ```json
   {
     "forecast": [442.1, 428.3, 440.6, 463.9, 481.4, 520.7, 585.1, 580.8, 513.5, 456.2, 410.8, 454.7],
     "index": ["1961-01-01", "1961-02-01", "1961-03-01", "1961-04-01", "1961-05-01", "1961-06-01", "1961-07-01", "1961-08-01", "1961-09-01", "1961-10-01", "1961-11-01", "1961-12-01"],
     "model_version": "1731196800",
     "metrics": {
       "mae": 16.4,
       "rmse": 20.8,
       "mape": 7.8
     }
   }
   ```

5. **Explore the companion notebook**

   Open `notebooks/arima_forecasting.ipynb` to inspect the dataset, re-run training, and visualise residual diagnostics.

6. **Optional Docker workflow**

   ```bash
   docker build -f fastapi_app/Dockerfile -t ml-fastapi .
   docker run --rm -p 8000:8000 ml-fastapi
   ```

   The container bundles every module. On the first forecasting request, the service trains automatically when artefacts are absent.

---

## Mathematical Foundations

ARIMA models difference a non-stationary series $d$ times, then fit an ARMA($p$, $q$) process to the resulting stationary sequence. The AR component captures auto-regressive structure

$$
X_t = \phi_1 X_{t-1} + \dots + \phi_p X_{t-p} + \varepsilon_t,
$$

while the MA component models serially correlated shock terms

$$
\varepsilon_t = \theta_1 u_{t-1} + \dots + \theta_q u_{t-q} + u_t.
$$

Combining differencing with AR and MA terms balances long-term trends and short-term fluctuations. Parameters $(p, d, q)$ are typically selected via information criteria or autocorrelation diagnostics; the default $(1,1,1)$ provides a solid baseline for AirPassengers.

### Plain-Language Intuition

The ARIMA model transforms monthly passenger counts into a nearly stationary series by subtracting each value from the previous month. It then learns how the differenced series depends on its own recent history and noise corrections. At prediction time, it rolls the equations forward step-by-step, translating differenced forecasts back into absolute passenger counts.

### When to Reach for ARIMA

- You have a single (univariate) time series with clear autocorrelation but no dominant seasonal pattern.
- Differencing once or twice makes the series look roughly stationary and residual diagnostics resemble white noise.
- You want an interpretable, fast baseline before moving to more complex state-space or machine-learning models.

### Strengths & Limitations

- **Strengths:** Transparent coefficients, excellent for short-term forecasts, handles stochastic trends with minimal data. Easy to automate with information criteria.
- **Limitations:** Struggles with strong seasonality (without seasonal terms), sudden structural breaks, or exogenous drivers unless extended to SARIMAX.

### Assumptions & Diagnostics

- Stationarity after differencing — check augmented Dickey–Fuller tests and rolling statistics.
- Residuals should be uncorrelated — inspect ACF/PACF of residuals and run Ljung–Box tests.
- Model order should balance fit vs. complexity — compare AIC/BIC scores across candidate $(p, d, q)$ grids.

---

## Dataset

- **Source**: Classic AirPassengers dataset (monthly airline passengers, 1949–1960).
- **Cache**: `data/air_passengers.csv` (bundled CSV).
- **Frequency**: Monthly (`MS`).

| Month       | Passengers |
|-------------|------------|
| 1949-01-01  | 112        |
| 1954-07-01  | 302        |
| 1960-12-01  | 432        |

The training routine keeps the chronological ordering intact and evaluates metrics on the final 15% of observations.

---

## Repository Layout

```
ARIMA/
├── README.md
├── demo.py
├── data/
│   └── air_passengers.csv
├── notebooks/
│   └── arima_forecasting.ipynb
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

- `config.py` centralises file paths, ARIMA order, and default forecast horizon.
- `data.py` loads the CSV, standardises the monthly index, and provides chronological splits.
- `pipeline.py` fits and evaluates the ARIMA model, persists artefacts via joblib/JSON, and exposes a reusable `forecast` method.
- `inference.py` defines the FastAPI schemas and a cached service that returns forecasts and stored metrics.
- `demo.py` offers a CLI sample to train and emit a 12-month projection.

---

## Extensions

1. **Order search** — grid-search $(p, d, q)$ combinations using AIC/BIC.
2. **Diagnostics** — integrate Ljung–Box tests and residual plots into the notebook.
3. **External regressors** — extend `pipeline.py` to accept exogenous variables when available.
4. **Rolling evaluation** — compute walk-forward metrics for production readiness.

---

## References

- George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung. *Time Series Analysis: Forecasting and Control*.
- Robert H. Shumway, David S. Stoffer. *Time Series Analysis and Its Applications*.
- statsmodels documentation: `statsmodels.tsa.arima.model.ARIMA`.
