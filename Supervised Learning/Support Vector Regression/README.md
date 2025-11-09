<!-- markdownlint-disable MD013 -->
# Support Vector Regression — California Housing Prices

This module packages a production-ready Support Vector Regression (SVR) workflow for predicting median house values across California block groups. It mirrors the pattern used by the other supervised-learning modules: clean configuration, reproducible training code, notebook exploration, persisted artefacts, FastAPI integration, and a lightweight demo script for quick experiments. Use it to revisit kernel regression fundamentals, prototype non-linear regressors, or showcase a tidy portfolio implementation.

---

## Learning Objectives

- Understand how SVR fits continuous targets by maximising the margin around a regression tube.
- Load and persist the California Housing dataset with consistent feature naming.
- Train, evaluate, and persist a StandardScaler + RBF-kernel SVR pipeline.
- Serve predictions through the shared FastAPI registry with typed request/response contracts.
- Extend the baseline with hyperparameter sweeps, calibration strategies, or monitoring hooks.

---

## Quickstart

1. **Install the shared environment**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model and persist artefacts**

   ```bash
   python "Supervised Learning/Support Vector Regression/src/train.py"
   ```

   Example output:

   ```json
   {
     "r2": 0.84,
     "rmse": 0.49,
     "mae": 0.36
   }
   ```

   The script writes `artifacts/svr_model.joblib` and `artifacts/metrics.json`. If `data/california_housing.csv` is missing, it is fetched from scikit-learn and cached locally.

3. **Launch the unified inference API**

   ```bash
   python -m fastapi_app.main
   ```

4. **Issue a prediction** (after registering the SVR slug in `fastapi_app/services/registry.py`; interactive docs are at `http://127.0.0.1:8000/docs`):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/support_vector_regression" \
        -H "Content-Type: application/json" \
        -d '{
              "median_income": 8.3252,
              "house_age": 41.0,
              "average_rooms": 6.9841,
              "average_bedrooms": 1.0238,
              "population": 322.0,
              "average_occupancy": 2.5556,
              "latitude": 37.88,
              "longitude": -122.23
            }'
   ```

   Sample response:

   ```json
   {
     "predicted_value": 4.74,
     "model_version": "1731196800",
     "metrics": {
       "r2": 0.84,
       "rmse": 0.49,
       "mae": 0.36
     }
   }
   ```

5. **Explore the companion notebook**

   Open `notebooks/svr.ipynb` to step through dataset exploration, train/validation parity checks, diagnostics, and kernel experiments.

6. **Run the API in Docker** (optional containerised workflow)

   ```bash
   docker build -f fastapi_app/Dockerfile -t ml-fastapi .
   docker run --rm -p 8000:8000 ml-fastapi
   ```

   The container bundles all supervised-learning services. On the first request the SVR service trains automatically if artefacts are missing.

---

## Mathematical Foundations

Support Vector Regression extends the maximum-margin idea to continuous targets. Given training pairs $(x_i, y_i)$, SVR finds a function $f(x) = w^\top \phi(x) + b$ that stays within an $\varepsilon$-wide tube around every target while keeping $w$ small:

$$
\min_{w, b, \xi, \xi^*} \frac{1}{2} \lVert w \rVert^2 + C \sum_i (\xi_i + \xi_i^*)
$$

subject to

$$
\begin{aligned}
y_i - f(x_i) &\leq \varepsilon + \xi_i, \\
 f(x_i) - y_i &\leq \varepsilon + \xi_i^*, \\
 \xi_i, \xi_i^* &\geq 0.
\end{aligned}
$$

Only samples that fall outside the tube (the support vectors) influence the solution. Kernel functions, such as the radial basis function (RBF) used here,

$$
K(x_i, x_j) = \exp\big(-\gamma \lVert x_i - x_j \rVert^2\big),
$$

allow the algorithm to learn non-linear relationships without explicitly mapping inputs to high-dimensional spaces.

### Plain-Language Intuition

Picture drawing a soft sleeve around the true house prices so that most predictions land inside the sleeve. Points that fall outside nudge the sleeve outward, but only in their neighbourhood. SVR tries to make that sleeve as thin as possible while still covering virtually all points, effectively balancing under- and over-shooting the real price.

When we use the RBF kernel, the model can bend this sleeve to follow curves in the data instead of staying perfectly straight. The parameters `C` and `epsilon` control how much the sleeve can bend and how tolerant it is of small errors. The reported metrics (R², RMSE, MAE) summarise how well the sleeve hugs unseen data drawn from the same distribution.

---

## Dataset

- **Source**: scikit-learn California Housing dataset (20,640 observations, 8 numeric predictors).
- **Location**: `data/california_housing.csv` (auto-downloaded on first run).
- **Target**: `median_house_value` (in $100k units).

Feature snapshot:

| Column              | Description                                   |
|---------------------|-----------------------------------------------|
| `median_income`     | Median income of households (scaled by 10k)   |
| `house_age`         | Median age of houses in the block             |
| `average_rooms`     | Average rooms per household                   |
| `average_bedrooms`  | Average bedrooms per household                |
| `population`        | Block group population                        |
| `average_occupancy` | Average household size                        |
| `latitude`          | Geographic latitude                           |
| `longitude`         | Geographic longitude                          |

The training routine performs an 80/20 split stratified by target quantiles to stabilise evaluation metrics over different runs.

---

## Repository Layout

```
Support Vector Regression/
├── README.md
├── data/
│   └── california_housing.csv    # Auto-downloaded dataset cache
├── notebooks/
│   └── svr.ipynb                 # Exploratory notebook mirroring src/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── pipeline.py
│   ├── train.py
│   └── inference.py
├── demo.py
└── artifacts/
    └── .gitkeep
```

---

## Implementation Walkthrough

### Code modules (`src/`)

- `config.py` centralises feature names, file paths, and train/test parameters while ensuring directories exist.
- `data.py` loads (or downloads) the dataset, normalises column names, and provides helper functions for `(X, y)` construction plus a quantile-stratified split.
- `pipeline.py` chains `StandardScaler` with an RBF-kernel `SVR`, trains the model, and logs R², RMSE, and MAE before persisting artefacts via joblib/JSON.
- `train.py` serves as the CLI entry point invoked in the Quickstart, returning the metrics dictionary for scripting.
- `inference.py` defines the Pydantic request/response schemas, caches a `CaliforniaHousingService`, lazily trains on-demand, and exposes `RequestModel`, `ResponseModel`, and `get_service()` for the FastAPI registry.

### Serving via FastAPI

After registering the slug in `fastapi_app/services/registry.py`, the shared API layer will:

- Validate incoming JSON against the `CaliforniaHousingRequest` schema.
- Load (or train) the persisted SVR pipeline and run predictions.
- Return the predicted house value alongside the stored metrics and a timestamp-based model version.

This matches the contract used by every supervised-learning module, keeping monitoring and deployment workflows uniform.

---

## Notebook Tour

`notebooks/svr.ipynb` walks through:

1. Dataset inspection and sanity checks against the production feature map.
2. Train/validation split parity to mirror the CLI pipeline.
3. Training, persistence, and evaluation of the SVR model.
4. Residual plots, error histograms, and kernel sensitivity experiments.
5. Notes on parameter tuning and monitoring ideas for future work.

Running the notebook after code changes assures parity between exploratory findings and the scripted pipeline.

---

## Extending the Module

1. **Hyperparameter sweeps** — grid-search over `C`, `epsilon`, and kernel types (e.g., linear, polynomial) to improve accuracy.
2. **Feature engineering** — add engineered features (e.g., income-to-house-age ratios) or leverage geospatial embeddings.
3. **Calibration** — analyse prediction intervals by bootstrapping residuals or wrapping SVR with conformal regressors.
4. **Monitoring** — log absolute error distributions over time to detect covariate drift in production data.
5. **Batch inference** — reuse `CaliforniaHousingService` for offline scoring jobs that write predictions back to data warehouses.

---

## References

- Alexander J. Smola, Bernhard Schölkopf. "A Tutorial on Support Vector Regression." *Statistics and Computing*, 2004.
- Christopher M. Bishop. *Pattern Recognition and Machine Learning.*
- scikit-learn documentation: `sklearn.svm.SVR` and the California housing dataset reference.
