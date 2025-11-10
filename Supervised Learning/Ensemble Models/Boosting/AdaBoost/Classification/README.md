<!-- markdownlint-disable MD013 -->
# AdaBoost Classification — Breast Cancer

This module packages an AdaBoost classifier for the scikit-learn breast cancer dataset. It follows the repository blueprint: declarative configuration, deterministic data caching, reproducible training, persisted artefacts, FastAPI integration, and a companion notebook slot. Use it to explain boosting with shallow base learners on a binary task.

---

## Learning Objectives

- Refresh boosting intuition using AdaBoost and shallow decision stumps.
- Cache the breast cancer dataset with consistent preprocessing and stratified splits.
- Train, evaluate, and persist an `AdaBoostClassifier` with tuned learning rate and estimator depth.
- Serve predictions through FastAPI with calibrated class probabilities and feature importances.
- Identify extension ideas such as early stopping, class weighting, or monitoring hooks.

---

## Quickstart

1. **Install shared requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the classifier**

   ```bash
   python "Supervised Learning/Ensemble Models/Boosting/AdaBoost/Classification/src/train.py"
   ```

   Sample metrics:

   ```json
   {
     "accuracy": 0.975,
     "precision": 0.973,
     "recall": 0.982,
     "f1": 0.978,
     "roc_auc": 0.993,
     "log_loss": 0.19
   }
   ```

   Artefacts saved to `artifacts/`:

   - `adaboost_classifier.joblib`
   - `metrics.json`
   - `feature_importances.json`

3. **Run the shared API**

   ```bash
   python -m fastapi_app.main
   ```

4. **POST a prediction** (`adaboost_classification` slug):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/adaboost_classification" \
        -H "Content-Type: application/json" \
        -d '{
              "mean_radius": 17.99,
              "mean_texture": 10.38,
              "mean_perimeter": 122.8,
              "mean_area": 1001.0,
              "mean_smoothness": 0.1184,
              "mean_compactness": 0.2776,
              "mean_concavity": 0.3001,
              "mean_concave_points": 0.1471,
              "mean_symmetry": 0.2419,
              "mean_fractal_dimension": 0.07871,
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
              "worst_perimeter": 184.6,
              "worst_area": 2019.0,
              "worst_smoothness": 0.1622,
              "worst_compactness": 0.6656,
              "worst_concavity": 0.7119,
              "worst_concave_points": 0.2654,
              "worst_symmetry": 0.4601,
              "worst_fractal_dimension": 0.1189
            }'
   ```

   Example response:

   ```json
   {
     "predicted_label": "malignant",
     "probability_of_malignant": 0.87,
     "class_probabilities": {
       "malignant": 0.87,
       "benign": 0.13
     },
     "model_version": "1731196800",
     "metrics": {
       "accuracy": 0.975,
       "precision": 0.973,
       "recall": 0.982,
       "f1": 0.978,
       "roc_auc": 0.993,
       "log_loss": 0.19
     },
     "feature_importances": {
       "worst_area": 0.11,
       "worst_perimeter": 0.09,
       "worst_radius": 0.09,
       "mean_concave_points": 0.07,
       "mean_radius": 0.06,
       "...": "..."
     }
   }
   ```

5. **Explore the notebook**

   Open `notebooks/adaboost_classification.ipynb` for residual analysis, staged accuracy curves, and parity checks with the scripted workflow.

---

## Foundations

AdaBoost iteratively reweights the training observations so that misclassified samples receive more emphasis in the next weak learner. Each learner contributes a weighted vote to the final prediction:

```
F_m(x) = F_{m-1}(x) + alpha_m * h_m(x)
```

Here `h_m` is the weak learner trained at iteration `m` and `alpha_m` controls its vote based on accuracy. With shallow decision trees this creates a powerful, low-variance ensemble that focuses on hard-to-classify observations.

### Plain-Language Intuition

Imagine repeatedly consulting junior analysts for second opinions. Each time they struggle with a tumour, you pay extra attention to that case in the next round. By the end, the aggregated opinions are strongly influenced by challenging examples, producing a robust final diagnosis.

---

## Dataset

- **Source**: scikit-learn Breast Cancer Wisconsin dataset (569 observations, 30 numeric predictors).
- **Caching**: `data/breast_cancer.csv` generated on first use for reproducibility.
- **Target**: `diagnosis` with labels `malignant` and `benign`.

The pipeline performs an 80/20 stratified split to maintain class balance.

---

## Repository Layout

```
Classification/
├── README.md
├── artifacts/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── demo.py
├── notebooks/
│   └── adaboost_classification.ipynb
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

- `config.py` defines hyperparameters, feature names, and deterministic paths.
- `data.py` downloads, normalises, and stratifies the breast cancer dataset.
- `pipeline.py` strings together scaling and `AdaBoostClassifier`, then logs metrics and feature importances.
- `inference.py` exposes typed request/response models and lazy loading for FastAPI inference.
- `demo.py` offers a quick CLI smoke test without spinning up the API layer.

---

## Extension Ideas

1. **Early stopping** — monitor staged scores and halt when validation accuracy plateaus.
2. **Class weighting** — adjust misclassification costs if ported to imbalanced datasets.
3. **Explainability** — add SHAP summaries or partial dependence plots in the notebook.
4. **Hyperparameter sweeps** — search over `n_estimators`, `learning_rate`, and base estimator depth.
5. **Monitoring** — push metrics to MLflow or your observability stack for production deployments.

---

## References

- Freund, Yoav; Schapire, Robert E. "A Short Introduction to Boosting." *Journal of Japanese Society for Artificial Intelligence*, 1999.
- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning.*
- scikit-learn documentation: `sklearn.ensemble.AdaBoostClassifier`.
