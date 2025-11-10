<!-- markdownlint-disable MD013 -->
# Gradient Boosting Regression — California Housing

This module operationalises a Gradient Boosting Regressor for the California housing dataset. It adheres to the repo’s production convention: deterministic config, tidy data prep, reproducible training, persisted artefacts, FastAPI inference, and an exploratory notebook. Leverage it to explain boosting for continuous targets and to compare against bagging/linear baselines.

---

## Learning Objectives

- Review gradient boosting as stage-wise additive modelling for regression.
- Cache and reuse the California housing dataset with quantile-aware splits.
- Train, evaluate, and persist a `GradientBoostingRegressor` with tuned learning rate and subsampling.
- Surface feature importances and staged learning curves via artefacts and API responses.
- Extend the baseline with early stopping, quantile loss, or monitoring hooks.

---

## Quickstart

1. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the regressor**

   ```bash
   python "Supervised Learning/Ensemble Models/Boosting/Gradient Boosting Machines/Regression/src/train.py"
   ```

   Example metrics:

   ```json
   {
     "r2": 0.89,
     "rmse": 0.41,
     "mae": 0.29
   }
   ```

   Artefacts saved to `artifacts/`:

   - `gradient_boosting_regressor.joblib`
   - `metrics.json`
   - `feature_importances.json`
   - `learning_curve.json`

3. **Serve via FastAPI**

   ```bash
   python -m fastapi_app.main
   ```

4. **POST a prediction** (`gradient_boosting_regression` slug):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/gradient_boosting_regression" \
        -H "Content-Type: application/json" \
        -d '{
              "median_income": 5.5,
              "house_age": 32.0,
              "average_rooms": 5.9,
              "average_bedrooms": 1.2,
              "population": 890.0,
              "average_occupancy": 2.7,
              "latitude": 36.8,
              "longitude": -121.9
            }'
   ```

   Response snapshot:

   ```json
   {
     "predicted_value": 4.62,
     "model_version": "1731196800",
     "metrics": {
       "r2": 0.89,
       "rmse": 0.41,
       "mae": 0.29
     },
     "feature_importances": {
       "median_income": 0.43,
       "latitude": 0.21,
       "longitude": 0.18,
       "house_age": 0.05,
       "average_rooms": 0.04,
       "average_bedrooms": 0.03,
       "population": 0.03,
       "average_occupancy": 0.03
     },
     "learning_curve": [
       {"iteration": 1, "rmse": 0.67},
       {"iteration": 50, "rmse": 0.51},
       {"iteration": 200, "rmse": 0.44},
       "..."
     ]
   }
   ```

5. **Interact with the notebook**

   Explore `notebooks/gradient_boosting_regression.ipynb` for staged loss visualisations, residual analysis, and parity checks with the CLI pipeline.

---

## Mathematical Foundations

Gradient boosting builds an additive model

$$
F_m(x) = F_{m-1}(x) + \nu h_m(x)
$$

where each weak learner \(h_m\) is fit to the negative gradient of the loss with respect to \(F_{m-1}(x)\). For squared error loss, this equates to fitting residuals at every stage. Shrinking the learning rate \(\nu\) while increasing the number of estimators reduces overfitting, and stochastic subsampling (`subsample < 1`) injects randomness akin to bagging while retaining the directional corrections of boosting.

### Plain-Language Intuition

Imagine iteratively refining a house-price spreadsheet: the first pass captures broad trends (income, location). Each subsequent pass focuses on the remaining errors, adding gentle corrections. Because every adjustment is small, the final forecast benefits from many subtle insights without veering wildly off course.

---

## Dataset

- **Source**: scikit-learn California housing dataset (20,640 observations, 8 numeric predictors).
- **Caching**: `data/california_housing.csv` created on-demand.
- **Target**: `median_house_value` (USD 100k units).

The training script performs an 80/20 split stratified by target quantiles to maintain representative price ranges across splits.

---

## Repository Layout

```
Regression/
├── README.md
├── artifacts/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── demo.py
├── notebooks/
│   └── gradient_boosting_regression.ipynb
└── src/
   ├── __init__.py
   ├── config.py
   ├── data.py
   ├── inference.py
   ├── pipeline.py
   └── train.py
```

---

## Implementation Highlights

- `config.py` stores feature names, hyperparameters, and file paths for deterministic runs.
- `data.py` downloads/cleans the dataset and generates quantile-aware train/validation splits.
- `pipeline.py` couples scaling with `GradientBoostingRegressor`, logs metrics, importances, and staged RMSE curves.
- `inference.py` delivers typed FastAPI schemas plus lazy model loading and auxiliary artefacts.
- `demo.py` provides a console sanity check without the API stack.

---

## Extension Ideas

1. **Early stopping** — track validation loss via `staged_predict` and halt when improvements plateau.
2. **Quantile loss** — switch `loss="quantile"` to model prediction intervals and expose them via the API.
3. **Feature engineering** — incorporate engineered ratios (rooms per household, income-age interactions) to capture non-linearities.
4. **Monitoring** — log learning curves and metrics to MLflow or your analytics stack for production observability.
5. **Hybrid ensembles** — blend gradient boosting with random forests or linear models for robust predictions.

---

## References

- Friedman, Jerome H. "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 2001.
- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning.*
- scikit-learn documentation: `sklearn.ensemble.GradientBoostingRegressor`.
