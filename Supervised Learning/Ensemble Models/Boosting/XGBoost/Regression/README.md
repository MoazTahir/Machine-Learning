<!-- markdownlint-disable MD013 -->
# XGBoost Regression — California Housing

This module operationalises an XGBoost regressor for the California housing dataset. It adheres to the repository blueprint: declarative configuration, deterministic data caching, reproducible training, persisted artefacts, FastAPI integration, and a notebook slot for exploration.

---

## Learning Objectives

- Apply XGBoost to a continuous target with shrinkage, subsampling, and regularisation.
- Cache the California housing dataset with quantile-aware train/validation splits.
- Train, evaluate, and persist an `XGBRegressor` while capturing staged RMSE metrics.
- Serve predictions via FastAPI complete with metrics, feature importances, and learning curves.
- Identify extension ideas such as early stopping, GPU acceleration, or hyperparameter sweeps.

---

## Quickstart

1. **Install shared requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the regressor**

   ```bash
   python "Supervised Learning/Ensemble Models/Boosting/XGBoost/Regression/src/train.py"
   ```

   Example metrics:

   ```json
   {
     "r2": 0.89,
     "rmse": 0.38,
     "mae": 0.27
   }
   ```

   Artefacts saved to `artifacts/`:

   - `xgboost_regressor.json`
   - `metrics.json`
   - `feature_importances.json`
   - `learning_curve.json`

3. **Run the shared API**

   ```bash
   python -m fastapi_app.main
   ```

4. **POST a prediction** (`xgboost_regression` slug):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/xgboost_regression" \
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
     "predicted_value": 4.66,
     "model_version": "1731196800",
     "metrics": {
       "r2": 0.89,
       "rmse": 0.38,
       "mae": 0.27
     },
     "feature_importances": {
       "median_income": 0.41,
       "latitude": 0.19,
       "longitude": 0.17,
       "average_rooms": 0.07,
       "house_age": 0.06,
       "average_bedrooms": 0.04,
       "population": 0.03,
       "average_occupancy": 0.03
     },
     "learning_curve": [
       {"iteration": 1, "rmse": 0.71},
       {"iteration": 100, "rmse": 0.44},
       {"iteration": 400, "rmse": 0.39},
       "..."
     ]
   }
   ```

5. **Explore the notebook**

   Open `notebooks/xgboost_regression.ipynb` for staged RMSE plots, feature importance visualisations, and parity checks with the scripted workflow.

---

## Foundations

XGBoost minimises a regularised objective that combines loss reduction with penalties on tree complexity. Successive trees are fit on the gradient of the loss, while shrinkage, subsampling, and column sampling keep variance in check. This yields strong predictive performance with manageable training times.

### Plain-Language Intuition

Picture an analyst who iteratively adds refined correction rules to a housing price model. Each rule must justify its benefit because overly complex additions are penalised. Subsampling encourages diverse insights, and shrinkage ensures every update remains measured.

---

## Dataset

- **Source**: scikit-learn California housing dataset (20,640 observations, eight numeric predictors).
- **Caching**: `data/california_housing.csv` generated on-demand for reproducibility.
- **Target**: `median_house_value` expressed in hundreds of thousands of USD.

An 80/20 split stratified by target quantiles maintains representative price ranges across train and validation sets.

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
│   └── xgboost_regression.ipynb
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

- `config.py` records features, hyperparameters, and artefact paths.
- `data.py` fetches and normalises the dataset, then prepares quantile-aware splits.
- `pipeline.py` trains `XGBRegressor`, logs metrics, feature importances, and evaluation history.
- `inference.py` exposes typed FastAPI schemas plus lazy model loading.
- `demo.py` offers a CLI smoke test without running the API.

---

## Extension Ideas

1. **Early stopping** — enable `early_stopping_rounds` with a validation set for automatic iteration control.
2. **GPU acceleration** — switch to `tree_method="gpu_hist"` when running on GPU-equipped hardware.
3. **Hyperparameter sweeps** — explore `max_depth`, `min_child_weight`, and `colsample_bytree` for optimal trade-offs.
4. **Explainability** — add SHAP summary plots or permutation importance to the notebook.
5. **Monitoring** — push metrics and learning curves to MLflow or another observability stack.

---

## References

- Chen, Tianqi; Guestrin, Carlos. "XGBoost: A Scalable Tree Boosting System." *KDD*, 2016.
- Janani, Harish. *Hands-On Gradient Boosting with XGBoost and scikit-learn.*
- xgboost documentation: `xgboost.XGBRegressor`.
