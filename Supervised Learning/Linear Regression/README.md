<!-- markdownlint-disable MD013 -->
# Linear Regression — Salary Prediction

This module is designed to be a complete learning path for simple linear regression: it introduces the math with intuition, demonstrates a scientific training workflow in Python, curates an exploratory notebook, and exposes the final model behind a production-ready FastAPI microservice. Use it to refresh fundamentals, teach others, or as a template for the rest of the repository.

---

## Learning Objectives

- Understand the assumptions and derivation of ordinary least squares (OLS).
- Explore and preprocess the salary dataset, motivating the need for scaling.
- Train, evaluate, and persist a regression pipeline using scikit-learn primitives.
- Expose the trained model through a modular inference service (FastAPI + optional Flask compatibility layer).
- Extend the notebook to experiment with regularisation, confidence intervals, and diagnostic plots.

---

## Quickstart

1. **Install the shared environment**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model and persist artefacts**

   ```bash
   python "Supervised Learning/Linear Regression/src/train.py"
   ```

   Example output:

   ```json
   {
     "r2": 0.978,
     "rmse": 4944.56,
     "mae": 3931.45
   }
   ```

   The script creates `artifacts/linear_regression_model.joblib` and `artifacts/metrics.json`.

3. **Launch the unified inference API**

   ```bash
   python -m fastapi_app.main
   ```

4. **Issue a prediction** (the FastAPI docs at `http://127.0.0.1:8000/docs` provide auto-generated forms):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/linear_regression" \
        -H "Content-Type: application/json" \
        -d '{"years_experience": 6.5}'
   ```

   Sample response:

   ```json
   {
     "salary_prediction": 93074.32,
     "model_version": "1730764800",
     "metrics": {
       "r2": 0.978,
       "rmse": 4944.56,
       "mae": 3931.45
     }
   }
   ```

5. **Deep dive via the notebook**

   Open `notebooks/linear_regression.ipynb` in VS Code or Jupyter to walk through the EDA, derivations, and diagnostics.

### Run the FastAPI service in Docker

If you prefer containerised workflows, build the pre-configured image and expose the API on port 8000:

```bash
docker build -f fastapi_app/Dockerfile -t ml-fastapi .
docker run --rm -p 8000:8000 ml-fastapi
```

The image bundles the repository and dependencies. On the first inference request the service will train the linear regression model automatically if the `artifacts/` directory is empty.

---

## Mathematical Foundations

Given a design matrix $X \in \mathbb{R}^{n \times p}$ and targets $y \in \mathbb{R}^{n}$, ordinary least squares (OLS) solves

$$
\hat{\beta} = \arg\min_{\beta} \lVert y - X\beta \rVert^2,
$$

producing the familiar closed-form solution (when $X^\top X$ is invertible)

$$
\hat{\beta} = (X^{\top}X)^{-1}X^{\top}y.
$$

Intuitively, OLS finds the straight line that minimises the squared vertical distance between observed salaries and predicted salaries. With one feature, the fitted line is

$$
\hat{y} = \beta_0 + \beta_1 x,
$$

where $\beta_1$ captures the expected salary increase per additional year of experience. For example, if $\beta_1 = 9{,}400$ and $\beta_0 = 29{,}000$, the model predicts a salary of roughly $29{,}000 + 9{,}400 \times 6.5 \approx 90{,}000$ USD for 6.5 years of experience.

### Assumptions worth checking

1. **Linearity** — residual plots should look like noise; polynomial trends would suggest feature engineering.
2. **Independence** — our dataset is a simple cross-section, but time-series problems would require specialised handling.
3. **Homoscedasticity** — variance of residuals should stay roughly constant across experience levels.
4. **Normality** — mainly relevant if you intend to form confidence intervals around coefficients.

The notebook includes code cells that generate Q-Q plots, residual vs. fitted charts, and leverage statistics to test these assumptions empirically.

---

## Dataset

- **Source**: classic salary vs. experience sample (30 observations).
- **Location**: `data/salary_data.csv`.
- **Schema**:
   - `YearsExperience` — continuous, measured in years.
   - `Salary` — annual salary in USD.

Preview:

| YearsExperience |   Salary |
|----------------:|---------:|
|             1.1 |  39343.0 |
|             3.2 |  64445.0 |
|             6.0 |  93940.0 |
|             9.5 | 116969.0 |

The training script performs a deterministic 80/20 split using `random_state=42`, enabling reproducible experiments and consistent metrics across notebook runs and CLI executions.

---

## Repository Layout

```
Linear Regression/
├── README.md                # You are here
├── data/                    # Local copy of the salary dataset
├── notebooks/               # Rich tutorial notebooks (see below)
├── src/
│   ├── config.py            # Central configuration dataclass
│   ├── data.py              # Data loading and preprocessing helpers
│   ├── pipeline.py          # Training pipeline and persistence helpers
│   ├── train.py             # CLI entry-point for training
│   └── inference.py         # FastAPI-ready inference service
└── artifacts/               # Model weights, metrics, and metadata
```

The `src/` code is importable and intentionally decoupled from notebooks or UI layers. The same abstraction pattern is repeated across all supervised-learning modules so you can compare architectures easily.

---

## Implementation Walkthrough

### Code modules (`src/`)

- `config.py` encapsulates all tunables (feature names, file paths, train/test ratio). It is a dataclass so you can override values during testing.
- `data.py` loads the CSV, builds the feature matrix/target vector, and performs the train/validation split. Every function accepts an optional config, making it easy to feed alternative datasets.
- `pipeline.py` constructs the scikit-learn pipeline, trains the model, and persists both artefacts and metrics via `joblib` and JSON.
- `train.py` is the CLI entry point. It calls `train_and_persist`, prints metrics as JSON, and is the script invoked in the Quickstart.
- `inference.py` defines Pydantic request/response models, a cached `LinearRegressionService`, and exposes `get_service()` so the FastAPI registry can bootstrap the endpoint lazily.

### Training pipeline (step-by-step)

1. **Dataset loading** — `train_validation_split` reads the configured file and splits according to `config.test_size`.
2. **Preprocessing** — numeric features are standardised with `StandardScaler` inside a `ColumnTransformer`. Keeping this logic in the pipeline guarantees identical preprocessing at inference time.
3. **Model fit** — scikit-learn’s `LinearRegression` runs on the scaled features.
4. **Evaluation** — metrics include:
   - `r2` (goodness of fit);
   - `rmse` (root mean squared error);
   - `mae` (mean absolute error).
5. **Persistence** — `pipeline.save()` writes `linear_regression_model.joblib`; `pipeline.write_metrics()` mirrors the metrics JSON used for documentation and monitoring.

Inspect `artifacts/metrics.json` after training to validate parity between notebook experiments and scripted training runs.

### Serving with FastAPI

The FastAPI project adopts a conventional structure:

- `core/config.py` — central settings (`app_name`, docs URLs, debug flags).
- `core/app.py` — builds the FastAPI instance and attaches routers.
- `models/health.py` — response schema for the health check.
- `services/registry.py` — declarative registry that discovers each ML module, loads its request/response models, and registers strongly-typed routes.
- `api/routes.py` — exposes `/health`, per-model endpoints (e.g. `/models/linear_regression`), and a dynamic `/models/{slug}/invoke` for programmatic routing.
- `main.py` — thin CLI wrapper around `uvicorn`.

Each algorithm module only needs to export `RequestModel`, `ResponseModel`, and `get_service()`. The registry takes care of wiring everything into the API.

---

## Notebook Tour

The companion notebook `notebooks/linear_regression.ipynb` mirrors the production code and adds rich commentary:

- **Cell 1–2**: primer on the maths with diagrams and intuition.
- **Cell 3**: imports that match the production pipeline exactly.
- **Cells 4–6**: dataset preview, scatter plots, distribution checks.
- **Cell 7**: training pipeline replicating `src/pipeline.py` to validate metrics inline.
- **Cell 8**: residual diagnostics to interrogate the assumptions discussed earlier.
- **Closing markdown**: instructions for syncing notebook experiments back into the codebase and for rolling changes into the FastAPI layer.

Running the notebook after each code change is a quick way to confirm that experiments and the scripted pipeline stay in lockstep.

---

## Extending the Module

1. **New datasets** — update `CONFIG.data_path` or instantiate `LinearRegressionConfig` with a custom path, then pass it into `train_and_persist(config)`.
2. **Regularisation** — replace `LinearRegression` with `Ridge`/`Lasso`, add hyperparameters to the config, and log them alongside metrics.
3. **Experiment tracking** — hook `train_and_persist` into MLflow or Weights & Biases to capture runs, or generate markdown model cards summarising metrics and assumptions.
4. **Cross-validation** — extend `pipeline.py` with `cross_val_score` or `GridSearchCV` to compare scoring strategies before persisting a model.
5. **Batch vs. online serving** — reuse `LinearRegressionService` in a CLI that loads `metrics.json` and prints predictions for CSV batches, showing how the same core objects can serve multiple delivery modes.

---

## References

- Freedman, D. A. *Statistical Models: Theory and Practice.*
- Hastie, T., Tibshirani, R., Friedman, J. *The Elements of Statistical Learning.*
- Montgomery, D., Peck, E., Vining, G. *Introduction to Linear Regression Analysis.*

Use these references while reading the notebook to connect statistical theory with the implementation.


