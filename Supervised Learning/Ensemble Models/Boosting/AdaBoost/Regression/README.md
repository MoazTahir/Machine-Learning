<!-- markdownlint-disable MD013 -->
# AdaBoost Regression — California Housing

This module delivers an AdaBoost regressor for the California housing dataset. It follows the repository conventions: declarative configuration, deterministic data caching, reproducible training, persisted artefacts, FastAPI integration, and an accompanying notebook slot.

---

## Learning Objectives

- Review boosting for continuous targets using AdaBoost with shallow decision trees.
- Cache the California housing dataset and reuse quantile-aware train/validation splits.
- Train, evaluate, and persist an `AdaBoostRegressor` with tuned learning rate and stage tracking.
- Serve predictions via FastAPI with exposed metrics, feature importances, and staged RMSE traces.
- Spot extension opportunities such as quantile loss, early stopping, or monitoring hooks.

---

## Quickstart

1. **Install shared requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the regressor**

   ```bash
   python "Supervised Learning/Ensemble Models/Boosting/AdaBoost/Regression/src/train.py"
   ```

   Example metrics:

   ```json
   {
     "r2": 0.83,
     "rmse": 0.47,
     "mae": 0.35
   }
   ```

   Artefacts saved to `artifacts/`:

   - `adaboost_regressor.joblib`
   - `metrics.json`
   - `feature_importances.json`
   - `learning_curve.json`

3. **Run the shared API**

   ```bash
   python -m fastapi_app.main
   ```

4. **POST a prediction** (`adaboost_regression` slug):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/adaboost_regression" \
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
     "predicted_value": 4.21,
     "model_version": "1731196800",
     "metrics": {
       "r2": 0.83,
       "rmse": 0.47,
       "mae": 0.35
     },
     "feature_importances": {
       "median_income": 0.39,
       "latitude": 0.18,
       "longitude": 0.16,
       "average_rooms": 0.08,
       "house_age": 0.07,
       "average_bedrooms": 0.05,
       "population": 0.04,
       "average_occupancy": 0.03
     },
     "learning_curve": [
       {"iteration": 1, "rmse": 0.71},
       {"iteration": 50, "rmse": 0.52},
       {"iteration": 200, "rmse": 0.46},
       "..."
     ]
   }
   ```

5. **Explore the notebook**

   Open `notebooks/adaboost_regression.ipynb` for staged loss visualisations, residual diagnostics, and parity checks with the scripted workflow.

---

## Foundations

AdaBoost regression builds an additive ensemble of shallow trees. Each stage fits the residuals from the current ensemble and contributes a weighted correction:

```
F_m(x) = F_{m-1}(x) + alpha_m * h_m(x)
```

Here `h_m` is the weak learner and `alpha_m` reflects its performance. By repeatedly focusing on residual error, the ensemble captures non-linear structure while keeping individual learners simple.

### Plain-Language Intuition

Think of a housing analyst who repeatedly revisits forecasts. Each pass concentrates on the remaining mistakes, adds a modest correction, and moves forward. After many iterations the aggregated corrections deliver a strong predictor that still respects the overall trend.

---

## Dataset

- **Source**: scikit-learn California housing dataset (20,640 observations, eight numeric predictors).
- **Caching**: `data/california_housing.csv` generated on-demand for reproducibility.
- **Target**: `median_house_value` expressed in hundreds of thousands of USD.

The pipeline uses an 80/20 split stratified by target quantiles to maintain representative price ranges.

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
│   └── adaboost_regression.ipynb
└── src/
   ├── __init__.py
   ├── config.py
   ├── data.py
   ├── inference.py
   ├── pipeline.py
   └── train.py
```

---

## Implementation Notes

- `config.py` captures feature names, hyperparameters, and artefact paths.
- `data.py` fetches and normalises the dataset, then prepares quantile-aware splits.
- `pipeline.py` chains scaling with `AdaBoostRegressor`, logging metrics, feature importances, and staged RMSE curves.
- `inference.py` exposes typed request/response objects with lazy model loading for FastAPI.
- `demo.py` provides a console smoke test without the API layer.

---

## Extension Ideas

1. **Quantile loss** — switch to `loss="linear"` or `loss="exponential"` to explore alternative optimisation criteria.
2. **Early stopping** — monitor staged RMSE and halt training once improvements plateau.
3. **Hybrid ensembles** — blend AdaBoost predictions with gradient boosting for ensembling experiments.
4. **Feature engineering** — add engineered ratios (rooms per household, latitude-longitude interactions) to capture non-linearities.
5. **Monitoring** — push metrics and learning curves to MLflow or your observability stack.

---

## References

- Drucker, Harris. "Improving Regressors using Boosting Techniques." *ICML*, 1997.
- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning.*
- scikit-learn documentation: `sklearn.ensemble.AdaBoostRegressor`.
