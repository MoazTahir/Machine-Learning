<!-- markdownlint-disable MD013 -->
# Exponential Smoothing — AirPassengers

This module implements Holt-Winters Exponential Smoothing for the AirPassengers dataset. It adheres to the repository’s production-style structure: configuration, scripted training, persisted artefacts, FastAPI integration, notebook exploration, and a CLI demo. Use it as a lightweight seasonal baseline alongside ARIMA, SARIMA, and Prophet.

---

## Learning Objectives

- Fit an additive-trend, multiplicative-seasonal Holt-Winters model.
- Evaluate forecasts against a chronological holdout slice.
- Deploy forecasts through the shared FastAPI registry.
- Compare smoothing-based performance against ARIMA-family and Prophet models.

---

## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and persist the model**

   ```bash
   python "Supervised Learning/Time Series Forcasting/Exponential Smoothing/src/train.py"
   ```

   Example output:

   ```json
   {
     "mae": 14.6,
     "rmse": 18.9,
     "mape": 6.8
   }
   ```

3. **Launch the FastAPI service**

   ```bash
   python -m fastapi_app.main
   ```

4. **Request a forecast**

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/exponential_smoothing_forecast" \
        -H "Content-Type: application/json" \
        -d '{"horizon": 12}'
   ```

5. **Open the notebook**

   `notebooks/exponential_smoothing.ipynb` mirrors the training pipeline and visualises forecast overlays.

---

## Mathematical Foundations

Holt-Winters smoothing maintains level ($L_t$), trend ($T_t$), and seasonal ($S_t$) components:

$$
\begin{aligned}
L_t &= \alpha \frac{y_t}{S_{t-s}} + (1 - \alpha)(L_{t-1} + T_{t-1}), \\
T_t &= \beta (L_t - L_{t-1}) + (1 - \beta) T_{t-1}, \\
S_t &= \gamma \frac{y_t}{L_t} + (1 - \gamma) S_{t-s},
\end{aligned}
$$

with forecasts given by

$$
\hat{y}_{t+h} = (L_t + h T_t) S_{t-s+h}.
$$

This additive trend + multiplicative seasonality configuration handles exponential growth while remaining computationally lightweight.

### Plain-Language Intuition

Think of Holt-Winters as three moving averages working together. One tracks the current level, one keeps tabs on how fast the series is growing, and the third memorises the typical seasonal bump for each month. Recent observations influence these components more than distant ones thanks to the exponential weights, so the model adapts quickly when patterns shift.

### When to Reach for Exponential Smoothing

- You need a fast, reliable baseline before exploring heavier ARIMA-family models.
- Seasonality is strong and stable, and the latest observations should influence forecasts more than older ones.
- You want a method that performs well with short histories and can be explained without heavy statistics.

### Strengths & Limitations

- **Strengths:** Extremely fast to train, adaptive to gradual changes, and easy to explain to stakeholders.
- **Limitations:** Assumes seasonal pattern repeats consistently; cannot handle multiple seasonalities; lacks built-in mechanisms for covariates or sudden structural breaks.

### Practical Tuning Tips

- Experiment with additive vs. multiplicative combinations for level, trend, and seasonality based on the data’s growth pattern.
- Enable damping (`damped_trend=True`) when long-range forecasts should flatten out.
- Compare smoothing levels (`alpha`, `beta`, `gamma`) across runs — higher values react faster but can chase noise.

### Diagnostics & Monitoring

- Plot fitted values vs. actuals to confirm seasonal alignment; misaligned peaks suggest the wrong seasonal period.
- Inspect residual autocorrelation — lingering spikes imply under-smoothed components or the need for ARIMA-style corrections.
- Track MAE/MAPE on a rolling window to detect drift; exponential smoothing adapts, but only if artefacts are refreshed regularly.

---

## Dataset

- **Source**: AirPassengers (monthly airline passengers, 1949–1960).
- **Cache**: `data/air_passengers.csv`.
- **Seasonality**: 12-month cycle.

---

## Repository Layout

```
Exponential Smoothing/
├── README.md
├── demo.py
├── data/
│   └── air_passengers.csv
├── notebooks/
│   └── exponential_smoothing.ipynb
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

- `config.py` — captures seasonal period, smoothing options, horizon, and file locations in a dataclass.
- `data.py` — loads the AirPassengers CSV, converts the `Month` column into a monthly index, and returns chronological splits for evaluation.
- `pipeline.py` — fits the Holt-Winters model via statsmodels, evaluates MAE/RMSE/MAPE, and persists weights plus metrics for reuse.
- `inference.py` — exposes the trained model to FastAPI with typed request/response schemas and cached loading semantics.
- `train.py` — serves as the scripted training entry point used in CI or manual retraining.
- `demo.py` — provides a quick, reproducible CLI example that prints a 12-month forecast to the console.

Each module mirrors the wider supervised-learning architecture so switching between algorithms is frictionless.

---

## Extensions

1. **Component tuning** — experiment with additive vs. multiplicative configurations.
2. **Damped trend** — enable damping for longer forecast horizons.
3. **Hybrid ensembles** — average forecasts from ARIMA/SARIMA/Prophet for robustness.

---

## References

- Charles C. Holt, "Forecasting Trends and Seasonals by Exponentially Weighted Moving Averages", 1957.
- Peter R. Winters, "Forecasting Sales by Exponentially Weighted Moving Averages", 1960.
- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*.
