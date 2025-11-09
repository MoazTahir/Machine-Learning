<!-- markdownlint-disable MD013 -->
# Support Vector Machine — Breast Cancer Classification

This module delivers a production-ready Support Vector Machine (SVM) pipeline for diagnosing malignant vs. benign breast tumours. It mirrors the structure used by the other supervised-learning modules: reusable configuration, scripted training, an exploratory notebook, persisted artefacts, and a FastAPI-compatible inference service. Use it as a reference implementation for high-dimensional classification problems with strong margin-based decision boundaries.

---

## Learning Objectives

- Refresh the geometric intuition behind maximum-margin classifiers and kernel tricks.
- Load, clean, and persist the UCI/Wisconsin breast cancer dataset in a reproducible way.
- Train, evaluate, and persist an SVM pipeline with feature scaling and class balancing.
- Expose the trained classifier behind the shared FastAPI service with typed request/response models.
- Extend the baseline with alternative kernels, hyperparameter searches, and monitoring hooks.

---

## Quickstart

1. **Install the shared environment**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model and persist artefacts**

   ```bash
   python "Supervised Learning/Support Vector Machine/src/train.py"
   ```

   Example output:

   ```json
   {
     "accuracy": 0.979,
     "precision": 0.981,
     "recall": 0.975,
     "f1": 0.978,
     "roc_auc": 0.997
   }
   ```

   The script writes `artifacts/svm_classifier.joblib` and `artifacts/metrics.json`. If `data/breast_cancer.csv` is missing, it is automatically downloaded from scikit-learn and cached locally.

3. **Launch the unified inference API**

   ```bash
   python -m fastapi_app.main
   ```

4. **Issue a prediction** (interactive docs available at `http://127.0.0.1:8000/docs` once the SVM slug is registered in `fastapi_app/services/registry.py`):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/support_vector_machine" \
        -H "Content-Type: application/json" \
        -d '{
              "mean_radius": 20.57,
              "mean_texture": 17.77,
              "mean_perimeter": 132.90,
              "mean_area": 1326.0,
              "mean_smoothness": 0.08474,
              "mean_compactness": 0.07864,
              "mean_concavity": 0.08690,
              "mean_concave_points": 0.07017,
              "mean_symmetry": 0.1812,
              "mean_fractal_dimension": 0.05667,
              "radius_error": 1.095,
              "texture_error": 0.9053,
              "perimeter_error": 8.589,
              "area_error": 153.4,
              "smoothness_error": 0.006399,
              "compactness_error": 0.04904,
              "concavity_error": 0.05373,
              "concave_points_error": 0.01587,
              "symmetry_error": 0.03003,
              "fractal_dimension_error": 0.006193,
              "worst_radius": 25.38,
              "worst_texture": 17.33,
              "worst_perimeter": 184.60,
              "worst_area": 2019.0,
              "worst_smoothness": 0.1622,
              "worst_compactness": 0.6656,
              "worst_concavity": 0.7119,
              "worst_concave_points": 0.2654,
              "worst_symmetry": 0.4601,
              "worst_fractal_dimension": 0.1189
            }'
   ```

   Sample response:

   ```json
   {
     "predicted_label": "malignant",
     "probability_malignant": 0.93,
     "model_version": "1731196800",
     "metrics": {
       "accuracy": 0.979,
       "precision": 0.981,
       "recall": 0.975,
       "f1": 0.978,
       "roc_auc": 0.997
     }
   }
   ```

5. **Explore the companion notebook**

   Open `notebooks/svm.ipynb` to step through dataset inspection, margin visualisation, hyperparameter experiments, and post-hoc diagnostics that mirror the scripted pipeline.

6. **Run the API in Docker** (optional containerised workflow)

   ```bash
   docker build -f fastapi_app/Dockerfile -t ml-fastapi .
   docker run --rm -p 8000:8000 ml-fastapi
   ```

   The container bundles all supervised-learning services. On the first inference request the SVM service will train automatically if artefacts are missing.

---

## Mathematical Foundations

Support Vector Machines seek the hyperplane that maximises the margin between classes. For a binary dataset $(x_i, y_i)$ with labels $y_i \in \{-1, 1\}$, the hard-margin formulation is:

$$
\min_{w, b} \frac{1}{2} \lVert w \rVert^2 \quad \text{s.t.} \quad y_i (w^\top x_i + b) \geq 1, \; \forall i.
$$

In practice we use the soft-margin variant with slack variables $\xi_i$ and penalty $C$:

$$
\min_{w, b, \xi} \frac{1}{2} \lVert w \rVert^2 + C \sum_i \xi_i \quad \text{s.t.} \quad y_i (w^\top x_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0.
$$

The dual problem admits kernel substitution, allowing us to operate in a high- (even infinite-) dimensional feature space without explicit mapping. This implementation uses the RBF kernel:

$$
K(x_i, x_j) = \exp\left(-\gamma \lVert x_i - x_j \rVert^2\right),
$$

which captures non-linear decision boundaries while retaining convex optimisation in the dual. Probabilities are calibrated by Platt scaling when `probability=True`.

### Assumptions and practices

1. **Feature scaling** — SVMs are sensitive to feature magnitudes, so the pipeline applies `StandardScaler` within the scikit-learn `Pipeline`.
2. **Margin maximisation** — Outliers can shrink the margin; tune `C` or experiment with robust preprocessing if heavy-tailed noise is observed.
3. **Kernel choice** — RBF is a sensible default; linear kernels work for high-dimensional text data, while polynomial/sigmoid kernels suit specialised niches.
4. **Class imbalance** — The dataset is slightly imbalanced; `class_weight="balanced"` rescales the penalty per class.

---

### Plain-Language Intuition

Picture drawing a loop around malignant tumours on a scatter plot so that the loop stays as far away as possible from every benign tumour. An SVM searches for that “widest possible gap” between the two groups. If the data can’t be separated cleanly in two dimensions, the RBF kernel quietly lifts the points into a higher-dimensional space where a clean separation is easier, then projects the answer back down for you.

Once the gap is found, new patient measurements simply check which side of the boundary they fall on. The extra probability step measures how confidently the point landed inside the malignant region: near the centre of the malignant side yields a high probability, while points near the boundary return values closer to 0.5, signalling uncertainty.

## Dataset

- **Source**: UCI / scikit-learn Wisconsin Diagnostic Breast Cancer dataset (569 samples, 30 continuous features).
- **Location**: `data/breast_cancer.csv` (auto-created on first run).
- **Target**: `diagnosis` with values `{malignant, benign}`. The positive class for metrics is `malignant`.

Feature snapshot:

| Column                  | Description                                  |
|-------------------------|----------------------------------------------|
| `mean_radius`           | Mean distance from centre to points on perimeter |
| `mean_texture`          | Standard deviation of grey-scale values      |
| `mean_concave_points`   | Mean number of concave portions of the contour|
| `radius_error`          | Standard error of radius measurements        |
| `worst_fractal_dimension` | "Worst" (mean of largest three) fractal dimension |

The training script performs an 80/20 stratified split so class proportions remain stable across folds.

---

## Repository Layout

```
Support Vector Machine/
├── README.md
├── data/
│   └── breast_cancer.csv        # Auto-downloaded, cached copy of the dataset
├── notebooks/
│   └── svm.ipynb                # Exploratory notebook mirroring src/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── pipeline.py
│   ├── train.py
│   └── inference.py
└── artifacts/
    └── .gitkeep
```

---

## Implementation Walkthrough

### Code modules (`src/`)

- `config.py` centralises file paths, feature names, target labelling, and train/test ratios, ensuring artefact directories exist.
- `data.py` loads the cached CSV (or downloads it from scikit-learn if absent), normalises column names, and provides helpers for `(X, y)` construction with stratified splits.
- `pipeline.py` composes `StandardScaler` with `SVC` (RBF kernel, probability calibration, balanced class weights), trains the model, evaluates accuracy/precision/recall/F1/ROC-AUC, and persists artefacts via joblib/JSON.
- `train.py` is the CLI entry point invoked in the Quickstart, returning the metrics dictionary for easy logging or scripting.
- `inference.py` defines the Pydantic request/response models, caches a `BreastCancerService`, lazily retrains if artefacts are missing, and exposes `RequestModel`, `ResponseModel`, and `get_service()` for the FastAPI registry.

### Serving via FastAPI

After registering the slug in `fastapi_app/services/registry.py`, the shared API layer will:

- Validate incoming JSON against the strongly typed `BreastCancerRequest` schema.
- Load the persisted pipeline (training on the fly if `artifacts/` is empty).
- Return the predicted label, malignancy probability, model version (file timestamp), and stored metrics in a consistent response payload.

This mirrors the integrations used for Linear Regression and Naive Bayes, enabling uniform observability and deployment tactics.

---

## Notebook Tour

`notebooks/svm.ipynb` walks through:

1. Dataset inspection and sanity checks against the production feature map.
2. Stratified splitting to confirm metrics parity with `src/train.py`.
3. Training and persistence of the SVM pipeline to replicate CLI behaviour.
4. Confusion matrix, ROC curve, and precision-recall diagnostics.
5. Exploration of support vectors, margin intuition, and kernel hyperparameters.

Running the notebook after pipeline changes is a quick way to validate regressions and document findings.

---

## Extending the Module

1. **Kernel search** — grid-search over `{'linear', 'rbf', 'poly'}` kernels and expose the best configuration through the config dataclass.
2. **Probability calibration** — evaluate `CalibratedClassifierCV` to tighten probability estimates for downstream decision-making.
3. **Feature selection** — integrate `SelectKBest` or recursive feature elimination to test sparsity-friendly margins.
4. **Monitoring** — log the distance to the decision boundary alongside predictions for drift detection.
5. **Batch inference** — reuse `BreastCancerService` in a CLI that scores CSV batches, matching the API contract offline.

---

## References

- Corinna Cortes, Vladimir Vapnik. "Support-vector networks." *Machine Learning*, 1995.
- Christopher M. Bishop. *Pattern Recognition and Machine Learning.*
- scikit-learn documentation: `sklearn.svm.SVC` and the breast cancer dataset reference.
