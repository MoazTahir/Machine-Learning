<!-- markdownlint-disable MD013 -->
# K-Nearest Neighbours — Diabetes Progression Regression

This module wraps a K-Nearest Neighbours (KNN) regressor for the scikit-learn diabetes dataset. It mirrors the repository's supervised-learning template by providing explicit configuration, scripted training, persisted artefacts, notebook exploration, FastAPI integration, and a lightweight demo script. Use it to explore distance-based regression, compare with linear baselines, or enrich the portfolio with a non-parametric regressor.

---

## Learning Objectives

- Revisit KNN for continuous targets, including neighbour weighting and distance metrics.
- Load and cache the diabetes dataset with consistent feature naming.
- Train, evaluate, and persist a `StandardScaler` + `KNeighborsRegressor` pipeline.
- Serve predictions via the shared FastAPI registry with typed request/response schemas.
- Extend the baseline with sensitivity analyses, error monitoring, or feature engineering.

---

## Quickstart

1. **Install shared dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and persist the model**

   ```bash
   python "Supervised Learning/K Nearest Neighbours/Regression/src/train.py"
   ```

   Example metrics output:

   ```json
   {
     "r2": 0.46,
     "rmse": 55.1,
     "mae": 43.2
   }
   ```

   Artefacts are saved to `artifacts/knn_regressor.joblib` and `artifacts/metrics.json`. `data/diabetes.csv` is generated automatically on the first run.

3. **Launch the FastAPI service**

   ```bash
   python -m fastapi_app.main
   ```

4. **Issue a prediction** (after registering the regression entry)

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/knn_regression" \
        -H "Content-Type: application/json" \
        -d '{
              "age": 0.0381,
              "sex": 0.0507,
              "bmi": 0.0617,
              "bp": 0.0219,
              "s1": -0.0442,
              "s2": -0.0348,
              "s3": -0.0434,
              "s4": -0.0026,
              "s5": 0.0199,
              "s6": -0.0176
            }'
   ```

   Sample response:

   ```json
   {
     "predicted_value": 154.2,
     "model_version": "1731196800",
     "metrics": {
       "r2": 0.46,
       "rmse": 55.1,
       "mae": 43.2
     }
   }
   ```

5. **Open the notebook**

   `notebooks/knn_regression.ipynb` reproduces the training pipeline, validation checks, and quick diagnostics.

6. **Optional Docker workflow**

   ```bash
   docker build -f fastapi_app/Dockerfile -t ml-fastapi .
   docker run --rm -p 8000:8000 ml-fastapi
   ```

   On the first regression request the service trains automatically if artefacts are absent.

---

## Mathematical Foundations

For regression, KNN predicts a continuous value by averaging the targets of the $k$ nearest neighbours:

$$
\hat{y}(x) = \frac{\sum_{x_i \in \mathcal{N}_k(x)} w_i \cdot y_i}{\sum_{x_i \in \mathcal{N}_k(x)} w_i},
$$

where $w_i = 1 / d(x, x_i)$ for distance-weighted voting. Using Minkowski distance (order 2) with standardised features keeps scales comparable across disparate clinical measurements. Choosing $k$ balances bias (large $k$) and variance (small $k$).

### Plain-Language Intuition

To estimate diabetes progression for a new patient, the model finds nine patients in the training set with the most similar clinical profiles. Closer matches contribute more weight thanks to distance-based weighting. Averaging their observed progression scores yields the prediction, with reported metrics (R², RMSE, MAE) summarising performance on a held-out validation slice.

---

## Dataset

- **Source**: scikit-learn diabetes dataset (442 samples, 10 normalised features, continuous target).
- **Cache**: `data/diabetes.csv` (auto-generated from `sklearn.datasets.load_diabetes`).
- **Target**: `disease_progression`, a quantitative measure collected one year after baseline.

Feature overview:

| Column | Description                                               |
|--------|-----------------------------------------------------------|
| `age`  | Age (normalised)                                         |
| `sex`  | Sex (normalised)                                         |
| `bmi`  | Body mass index                                          |
| `bp`   | Average blood pressure                                   |
| `s1`   | Total serum cholesterol (TC)                              |
| `s2`   | Low-density lipoproteins (LDL)                            |
| `s3`   | High-density lipoproteins (HDL)                           |
| `s4`   | Thyroid stimulating hormone (TCH)                         |
| `s5`   | LTG (a blood serum measurement)                           |
| `s6`   | Blood sugar level (GLU)                                   |

---

## Repository Layout

```
Regression/
├── README.md
├── data/
│   └── diabetes.csv               # Cached dataset (auto-generated)
├── artifacts/
│   └── .gitkeep
├── notebooks/
│   └── knn_regression.ipynb       # Exploratory notebook mirroring src/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── pipeline.py
│   ├── train.py
│   └── inference.py
└── demo.py
```

---

## Implementation Walkthrough

- `config.py` centralises feature ordering, neighbour hyperparameters, and artefact paths.
- `data.py` downloads, caches, and exposes helpers for train/validation splits.
- `pipeline.py` chains scaling with `KNeighborsRegressor`, logs regression metrics, and persists artefacts to disk.
- `train.py` exposes the CLI hook used in automation and documentation.
- `inference.py` defines the FastAPI request/response schemas and a cached service.
- `demo.py` provides a quick CLI probe for experimentation.

---

## Extensions

1. **Hyperparameter tuning** — search over `n_neighbors`, `weights`, and `metric` choices.
2. **Feature engineering** — derive interaction terms or revert normalisation using domain ranges.
3. **Uncertainty estimates** — report neighbour variance or bootstrap confidence intervals.
4. **Monitoring** — log residual distributions to spot covariate drift in production data.
5. **Batch inference** — adapt `DiabetesRegressionService` for offline scoring jobs.

---

## References

- Trevor Hastie, Robert Tibshirani, Jerome Friedman. *The Elements of Statistical Learning*.
- Richard O. Duda, Peter E. Hart, David G. Stork. *Pattern Classification*.
- scikit-learn documentation: `sklearn.datasets.load_diabetes`, `sklearn.neighbors.KNeighborsRegressor`.
