<!-- markdownlint-disable MD013 -->
# Random Forest Regression — California Housing

This module wraps a Random Forest Regressor for predicting median house values across California block groups. It matches the repository’s supervised-learning template with deterministic config, reproducible data prep, a persisted pipeline, FastAPI integration, and an exploratory notebook. Use it to showcase bagging for continuous targets and to compare against single-tree or boosting baselines.

---

## Learning Objectives

- Understand how bootstrap aggregation stabilises tree-based regressors on noisy, heterogeneous data.
- Persist the California Housing dataset, apply quantile-aware splits, and reproduce metrics across runs.
- Train, evaluate, and store a `RandomForestRegressor` with 400 estimators and sensible regularisation defaults.
- Surface feature importances and metrics through the shared FastAPI registry for downstream monitoring.
- Extend the baseline with hyperparameter searches, partial dependence plots, or batch scoring utilities.

---

## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the regressor**

   ```bash
   python "Supervised Learning/Ensemble Models/Bagging/Random Forest/Regression/src/train.py"
   ```

   Example output:

   ```json
   {
     "r2": 0.86,
     "rmse": 0.43,
     "mae": 0.31,
     "num_trees": 400.0,
     "max_depth": -1.0
   }
   ```

   Artefacts saved to `artifacts/`:

   - `random_forest_regressor.joblib`
   - `metrics.json`
   - `feature_importances.json`

3. **Serve via FastAPI**

   ```bash
   python -m fastapi_app.main
   ```

4. **Issue a prediction** (endpoint registered as `random_forest_regression`):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/random_forest_regression" \
        -H "Content-Type: application/json" \
        -d '{
              "median_income": 6.32,
              "house_age": 34.0,
              "average_rooms": 5.4,
              "average_bedrooms": 1.1,
              "population": 980.0,
              "average_occupancy": 2.8,
              "latitude": 37.88,
              "longitude": -122.23
            }'
   ```

   Response snippet:

   ```json
   {
     "predicted_value": 4.82,
     "model_version": "1731196800",
     "metrics": {
       "r2": 0.86,
       "rmse": 0.43,
       "mae": 0.31
     },
     "feature_importances": {
       "median_income": 0.45,
       "latitude": 0.20,
       "longitude": 0.18,
       "house_age": 0.04,
       "average_rooms": 0.04,
       "average_bedrooms": 0.03,
       "population": 0.03,
       "average_occupancy": 0.03
     }
   }
   ```

5. **Inspect the notebook**

   Open `notebooks/random_forest_regression.ipynb` for exploratory analysis, residual diagnostics, and feature importance plots.

6. **Docker workflow** *(optional)*

   ```bash
   docker build -f fastapi_app/Dockerfile -t ml-fastapi .
   docker run --rm -p 8000:8000 ml-fastapi
   ```

---

## Mathematical Foundations

Random forest regression averages the predictions of \(B\) decision trees, each fit on a bootstrap sample with feature subsampling. For an input \(x\), the prediction is:

$$
\hat{f}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x),
$$

where \(T_b(x)\) is the response from the \(b\)-th tree. Bagging smooths out high-variance behaviour—individual trees may overfit, but their average converges to a low-variance estimator as \(B\) grows. For noisy regression tasks, this stabilisation often yields strong performance without heavy tuning.

### Plain-Language Intuition

Think of many survey teams estimating house prices. Each team samples different neighbourhoods and focuses on a random subset of features (like location, rooms, or age). Even if one team overestimates a particular area, most others will disagree, so the average remains reliable. The forest’s feature importances highlight which attributes the teams relied on most when forming their estimates.

---

## Dataset

- **Source**: scikit-learn California Housing dataset (20,640 records, 8 numeric predictors).
- **Caching**: `data/california_housing.csv` auto-generated on first run.
- **Target**: `median_house_value` (in units of $100k).

| Feature             | Description                                      |
|---------------------|--------------------------------------------------|
| `median_income`     | Median income in the block (scaled by $10k).      |
| `latitude`          | Geographic latitude.                              |
| `longitude`         | Geographic longitude.                             |
| `average_rooms`     | Average rooms per household.                      |
| `average_occupancy` | Average household size.                           |

The training routine performs an 80/20 split stratified by target quantiles so that both train and validation sets capture similar price ranges.

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
│   └── random_forest_regression.ipynb
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

- `config.py` centralises feature names, hyperparameters, and artefact paths for deterministic runs.
- `data.py` fetches the dataset, cleans column names, persists the CSV, and performs quantile-aware splits.
- `pipeline.py` wires a scaler with `RandomForestRegressor`, logs metrics, and records feature importances.
- `train.py` exposes a CLI for reproducible training in automation workflows.
- `inference.py` provides typed request/response models and lazy model loading for the FastAPI registry.
- `demo.py` sanity-checks predictions without bringing up the API layer.

---

## Extension Ideas

1. **Hyperparameter optimisation** — tune `max_depth`, `min_samples_leaf`, and `max_features` using random search or Bayesian optimisation.
2. **Quantile forests** — switch to `RandomForestQuantileRegressor` (from extra libraries) to return prediction intervals.
3. **Hybrid models** — average the random forest with gradient boosting or linear baselines to improve stability.
4. **Explainability** — integrate permutation importance or SHAP for richer interpretability.
5. **Batch inference** — build a CLI that streams large CSV batches through the service while logging metrics.

---

## References

- Leo Breiman. "Random Forests." *Machine Learning*, 2001.
- Friedman, Hastie, Tibshirani. *The Elements of Statistical Learning.*
- scikit-learn documentation: `sklearn.ensemble.RandomForestRegressor`.
