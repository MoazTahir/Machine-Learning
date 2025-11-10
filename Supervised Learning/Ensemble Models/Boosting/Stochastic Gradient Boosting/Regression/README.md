<!-- markdownlint-disable MD013 -->
# Stochastic Gradient Boosting Regression — California Housing

This module delivers a stochastic gradient boosting regressor for the California housing dataset. It mirrors the repository blueprint: declarative configuration, deterministic data caching, reproducible training, persisted artefacts, FastAPI integration, and a companion notebook. The focus here is on the variance-reduction benefits of subsampling within gradient boosting.

---

## Learning Objectives

- Refresh gradient boosting for continuous targets with emphasis on stochastic subsampling.
- Cache the California housing dataset and reuse quantile-aware train/validation splits.
- Train, evaluate, and persist a `GradientBoostingRegressor` configured with subsampling and feature sampling.
- Serve predictions via FastAPI while exposing metrics, feature importances, and staged learning curves.
- Identify natural extensions such as early stopping, quantile loss, or experiment tracking integrations.

---

## Quickstart

1. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the regressor**

   ```bash
   python "Supervised Learning/Ensemble Models/Boosting/Stochastic Gradient Boosting/Regression/src/train.py"
   ```

   Example metrics:

   ```json
   {
     "r2": 0.90,
     "rmse": 0.39,
     "mae": 0.28
   }
   ```

   Artefacts saved to `artifacts/`:

   - `stochastic_gradient_boosting_regressor.joblib`
   - `metrics.json`
   - `feature_importances.json`
   - `learning_curve.json`

3. **Serve via FastAPI**

   ```bash
   python -m fastapi_app.main
   ```

4. **POST a prediction** (`stochastic_gradient_boosting_regression` slug):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/stochastic_gradient_boosting_regression" \
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
       "predicted_value": 4.58,
       "model_version": "1731196800",
       "metrics": {
          "r2": 0.90,
          "rmse": 0.39,
          "mae": 0.28
       },
       "feature_importances": {
          "median_income": 0.42,
          "latitude": 0.20,
          "longitude": 0.17,
          "house_age": 0.06,
          "average_rooms": 0.05,
          "average_bedrooms": 0.04,
          "population": 0.03,
          "average_occupancy": 0.03
       },
       "learning_curve": [
          {"iteration": 1, "rmse": 0.64},
          {"iteration": 75, "rmse": 0.45},
          {"iteration": 200, "rmse": 0.41},
          "..."
       ]
    }
    ```

5. **Explore the notebook**

   Open `notebooks/stochastic_gradient_boosting_regression.ipynb` for staged loss visualisations, residual diagnostics, and parity checks with the scripted workflow.

---

## Foundations

Stochastic gradient boosting fits shallow trees in sequence, each correcting the residual error of the current ensemble. The update pattern is shown below:

```
F_0(x) = baseline
F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
```

Each weak learner `h_m` is trained on the negative gradient of the loss function—for squared error this reduces to fitting residuals. By sampling rows (`subsample < 1.0`) and features (`max_features="sqrt"`) at each iteration, the trees become less correlated, delivering regularisation similar to bagging whilst keeping the directed corrections of boosting.

### Plain-Language Intuition

Picture a pricing analyst who repeatedly revisits housing data. Every pass samples a different subset of neighbourhoods and features, adds a modest correction, then moves on. The learning rate keeps adjustments small and measured, while subsampling prevents any single pocket of data from dominating the outcome.

---

## Dataset

- **Source**: scikit-learn California housing dataset (20,640 observations, eight numeric predictors).
- **Caching**: `data/california_housing.csv` generated on-demand for reproducibility.
- **Target**: `median_house_value` (expressed in hundreds of thousands of USD).

The pipeline conducts an 80/20 split stratified by target quantiles so each fold represents the price distribution.

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
│   └── stochastic_gradient_boosting_regression.ipynb
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
- `pipeline.py` couples scaling with `GradientBoostingRegressor`, applies subsampling, and logs metrics, importances, plus staged RMSE curves.
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
