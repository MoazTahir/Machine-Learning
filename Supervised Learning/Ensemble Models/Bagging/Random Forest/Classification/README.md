<!-- markdownlint-disable MD013 -->
# Random Forest — Breast Cancer Classification

This module delivers a production-ready Random Forest pipeline for diagnosing malignant vs. benign breast tumours. It mirrors the repository-wide pattern: deterministic config, scripted training, persisted artefacts, FastAPI-ready inference, and a companion notebook. Use it to demonstrate bagging intuition, feature importance reporting, and robust performance on medium tabular datasets.

---

## Learning Objectives

- Refresh how bootstrap aggregation (bagging) reduces variance for tree-based models.
- Load, cache, and prepare the scikit-learn Breast Cancer dataset with reproducible transforms.
- Train, evaluate, and persist a `RandomForestClassifier` with calibrated class weights and 300 estimators.
- Serve predictions through the shared FastAPI registry with typed payloads and importances surfaced.
- Extend the baseline with hyperparameter searches, monotonic constraints, or monitoring hooks.

---

## Quickstart

1. **Install shared dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and persist artefacts**

   ```bash
   python "Supervised Learning/Ensemble Models/Bagging/Random Forest/Classification/src/train.py"
   ```

   Sample metrics:

   ```json
   {
     "accuracy": 0.982,
     "macro_precision": 0.981,
     "macro_recall": 0.982,
     "macro_f1": 0.981,
     "roc_auc": 0.998,
     "num_trees": 300.0,
     "max_depth": -1.0
   }
   ```

   Artefacts written to `artifacts/`:

   - `random_forest_classifier.joblib` — persisted pipeline (scaler + estimator).
   - `metrics.json` — evaluation dictionary for dashboards.
   - `feature_importances.json` — ranked feature importance vector.

3. **Run the API locally**

   ```bash
   python -m fastapi_app.main
   ```

4. **Send a prediction** (after registering the slug in `fastapi_app/services/registry.py`):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/random_forest_classification" \
        -H "Content-Type: application/json" \
        -d '{
              "mean_radius": 14.05,
              "mean_texture": 14.96,
              "mean_perimeter": 92.25,
              "mean_area": 606.5,
              "mean_smoothness": 0.09711,
              "mean_compactness": 0.06154,
              "mean_concavity": 0.01981,
              "mean_concave_points": 0.01768,
              "mean_symmetry": 0.1837,
              "mean_fractal_dimension": 0.05936,
              "radius_error": 0.5437,
              "texture_error": 0.7339,
              "perimeter_error": 3.398,
              "area_error": 51.32,
              "smoothness_error": 0.005225,
              "compactness_error": 0.01855,
              "concavity_error": 0.01988,
              "concave_points_error": 0.00653,
              "symmetry_error": 0.02045,
              "fractal_dimension_error": 0.003582,
              "worst_radius": 15.3,
              "worst_texture": 19.35,
              "worst_perimeter": 102.5,
              "worst_area": 729.8,
              "worst_smoothness": 0.1347,
              "worst_compactness": 0.114,
              "worst_concavity": 0.07081,
              "worst_concave_points": 0.03354,
              "worst_symmetry": 0.230,
              "worst_fractal_dimension": 0.07699
            }'
   ```

   Response excerpt:

   ```json
   {
     "predicted_label": "benign",
     "probability_of_malignant": 0.03,
     "model_version": "1731196800",
     "metrics": {
       "accuracy": 0.982,
       "macro_precision": 0.981,
       "macro_recall": 0.982,
       "macro_f1": 0.981,
       "roc_auc": 0.998
     },
     "feature_importances": {
       "worst_radius": 0.112,
       "mean_texture": 0.081,
       "worst_concave_points": 0.068,
       "worst_area": 0.064,
       "worst_perimeter": 0.061,
       "...": "..."
     }
   }
   ```

5. **Explore the notebook**

   Launch `notebooks/random_forest_classification.ipynb` for exploratory analysis, SHAP-style interpretation hooks, and reproducible experiments mirroring the CLI pipeline.

6. **Optional Dockerised serving**

   ```bash
   docker build -f fastapi_app/Dockerfile -t ml-fastapi .
   docker run --rm -p 8000:8000 ml-fastapi
   ```

---

## Mathematical Foundations

Random forests average the predictions from many decision trees trained on bootstrap samples. For classification, the forest produces class probabilities as the proportion of trees voting for each class. Bagging reduces variance: while a single tree may overfit idiosyncrasies in the training set, the ensemble de-correlates errors by injecting randomness via bootstrap sampling and feature subsampling (`max_features=sqrt`).

The probability estimate for class \(c\) is:

$$
\hat{P}(y = c \mid x) = \frac{1}{B} \sum_{b=1}^{B} \mathbb{1}\{T_b(x) = c\},
$$

where \(B\) is the number of trees, and \(T_b\) is the prediction of the \(b\)-th tree. As \(B \to \infty\), the law of large numbers guarantees convergence of the ensemble to its expectation, stabilising predictions.

### Plain-Language Intuition

Imagine handing the same patient record to 300 oncologists who each train on slightly different historical samples and focus on a random subset of measurements. Each doctor casts a vote—most still agree on clear-cut cases, while borderline examples trigger debate. By aggregating their votes, the random forest produces a consensus diagnosis that is far less noisy than any single doctor’s opinion.

---

## Dataset

- **Source**: scikit-learn Breast Cancer Wisconsin Diagnostic dataset (569 samples, 30 continuous predictors).
- **Caching**: `data/breast_cancer.csv` is created on first run for reproducibility.
- **Target**: `diagnosis` with labels `malignant` and `benign`.

| Column                 | Description                                             |
|------------------------|---------------------------------------------------------|
| `mean_radius`          | Mean distance from centre to perimeter points          |
| `worst_concave_points` | Largest concave portions of the contour                |
| `radius_error`         | Standard error of radius measurements                  |
| `worst_area`           | Mean of the three largest values of area               |
| `mean_fractal_dimension` | Fractal dimension of the tumour contour             |

Stratified 80/20 splits preserve class balance across training and validation sets.

---

## Repository Layout

```
Classification/
├── README.md
├── artifacts/
│   ├── .gitkeep
│   ├── feature_importances.json
│   └── random_forest_classifier.joblib
├── data/
│   ├── .gitkeep
│   └── breast_cancer.csv        # auto-downloaded cache
├── demo.py
├── notebooks/
│   └── random_forest_classification.ipynb
└── src/
   ├── __init__.py
   ├── config.py
   ├── data.py
   ├── inference.py
   ├── pipeline.py
   └── train.py
```

---

## Implementation Walkthrough

- `config.py` — centralises feature names, hyperparameters, and artefact locations.
- `data.py` — fetches the dataset from scikit-learn, normalises column names, persists a cached CSV, and supplies train/validation splits.
- `pipeline.py` — stitches a standard scaler with a `RandomForestClassifier`, logs metrics, and persists both the model and feature importances.
- `train.py` — CLI entrypoint used in the quickstart; prints metrics as JSON for automation pipelines.
- `inference.py` — exposes typed Pydantic schemas, lazy model loading, and probability-calibrated responses for the FastAPI registry.
- `demo.py` — lightweight script to sanity-check predictions without spinning up the API.

---

## Extending the Module

1. **Hyperparameter tuning** — run `sklearn.model_selection.RandomizedSearchCV` over tree depth, `max_features`, and `min_samples_leaf` to squeeze out extra accuracy.
2. **Probability calibration** — wrap the forest in `CalibratedClassifierCV` if you need well-calibrated probabilities for decision support.
3. **Interpretability** — integrate SHAP or permutation importance to expose global and local explanations via the FastAPI response payload.
4. **Class imbalance** — experiment with SMOTE or tweak `class_weight` if integrating with datasets exhibiting severe skew.
5. **Monitoring** — log vote distributions or out-of-bag error when shipping to production to detect drift.

---

## References

- Leo Breiman. "Random Forests." *Machine Learning*, 2001.
- Géron, Aurélien. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.
- scikit-learn documentation: `sklearn.ensemble.RandomForestClassifier`.
