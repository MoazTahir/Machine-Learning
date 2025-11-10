<!-- markdownlint-disable MD013 -->
# XGBoost Classification — Wine Recognition

This module packages an XGBoost classifier for the UCI wine dataset. It mirrors the repository blueprint: declarative configuration, deterministic data caching, reproducible training, persisted artefacts, FastAPI integration, and a companion notebook slot. Use it to showcase gradient boosting with tree-based learners enhanced by second-order optimisation.

---

## Learning Objectives

- Understand XGBoost’s regularised objective and tree boosting workflow.
- Cache and reuse the wine dataset with deterministic preprocessing and stratified splits.
- Train, evaluate, and persist an `XGBClassifier` with tuned learning rate, depth, and sampling.
- Serve predictions via FastAPI with probability outputs and feature importances.
- Identify practical extensions such as early stopping, GPU acceleration, or hyperparameter sweeps.

---

## Quickstart

1. **Install shared requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the classifier**

   ```bash
   python "Supervised Learning/Ensemble Models/Boosting/XGBoost/Classification/src/train.py"
   ```

   Sample metrics:

   ```json
   {
     "accuracy": 0.982,
     "macro_precision": 0.983,
     "macro_recall": 0.980,
     "macro_f1": 0.981,
     "log_loss": 0.13,
     "roc_auc": 0.999
   }
   ```

   Artefacts saved to `artifacts/`:

   - `xgboost_classifier.json`
   - `metrics.json`
   - `feature_importances.json`

3. **Run the shared API**

   ```bash
   python -m fastapi_app.main
   ```

4. **POST a prediction** (`xgboost_classification` slug):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/xgboost_classification" \
        -H "Content-Type: application/json" \
        -d '{
              "alcohol": 13.2,
              "malic_acid": 1.78,
              "ash": 2.14,
              "alcalinity_of_ash": 11.2,
              "magnesium": 100.0,
              "total_phenols": 2.65,
              "flavanoids": 2.76,
              "nonflavanoid_phenols": 0.26,
              "proanthocyanins": 1.28,
              "color_intensity": 4.38,
              "hue": 1.05,
              "od280_od315_of_diluted_wines": 3.4,
              "proline": 1050.0
            }'
   ```

   Example response:

   ```json
   {
     "predicted_label": "class_0",
     "class_probabilities": {
       "class_0": 0.92,
       "class_1": 0.05,
       "class_2": 0.03
     },
     "model_version": "1731196800",
     "metrics": {
       "accuracy": 0.982,
       "macro_precision": 0.983,
       "macro_recall": 0.980,
       "macro_f1": 0.981,
       "log_loss": 0.13,
       "roc_auc": 0.999
     },
     "feature_importances": {
       "color_intensity": 0.15,
       "proline": 0.13,
       "flavanoids": 0.11,
       "alcohol": 0.10,
       "od280_od315_of_diluted_wines": 0.09,
       "...": "..."
     }
   }
   ```

5. **Explore the notebook**

   Open `notebooks/xgboost_classification.ipynb` for confusion matrices, feature importance plots, and parity checks with the scripted workflow.

---

## Foundations

XGBoost extends gradient boosting with second-order Taylor approximations, shrinkage, column subsampling, and regularisation terms. Each tree is fit to minimise a regularised objective that balances loss reduction against model complexity:

```
Obj = \sum_i l(y_i, \hat{y}_i^{(t)}) + \sum_k \Omega(f_k)
```

The regulariser `\Omega` penalises tree depth and leaf weights, helping the model generalise while training quickly through efficient column sampling and parallel tree construction.

### Plain-Language Intuition

Imagine a committee of specialists incrementally refining their wine classification rules. Each new specialist considers how previous mistakes can be corrected, but they are also penalised for adding unnecessarily complex rules. Subsampling ensures diversity, while shrinkage keeps every update measured.

---

## Dataset

- **Source**: UCI Wine dataset (178 samples, 13 continuous predictors across three cultivars).
- **Caching**: `data/wine.csv` generated on first run for reproducibility.
- **Target**: `class_label` with values `class_0`, `class_1`, `class_2`.

An 80/20 stratified split maintains class balance, and a fixed random seed ensures deterministic results.

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
│   └── xgboost_classification.ipynb
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

- `config.py` defines features, hyperparameters, and deterministic paths.
- `data.py` fetches and normalises the wine dataset, persisting a cached CSV.
- `pipeline.py` trains `XGBClassifier`, captures metrics, and dumps feature importances with label encoding.
- `inference.py` loads the persisted model plus encoder to return probability-weighted predictions through FastAPI.
- `demo.py` offers a CLI smoke test without deploying the API layer.

---

## Extension Ideas

1. **Early stopping** — supply a validation set to `fit` with `early_stopping_rounds` for automatic iteration control.
2. **GPU acceleration** — enable `tree_method="gpu_hist"` where appropriate for faster training.
3. **Hyperparameter sweeps** — grid search over `max_depth`, `learning_rate`, and `subsample` to benchmark accuracy vs. latency.
4. **Explainability** — add SHAP value plots or permutation importance to the notebook.
5. **Monitoring** — log evaluation metrics to MLflow or your observability stack.

---

## References

- Chen, Tianqi; Guestrin, Carlos. "XGBoost: A Scalable Tree Boosting System." *KDD*, 2016.
- Nishant Shukla. *Machine Learning with XGBoost and scikit-learn.*
- xgboost documentation: `xgboost.XGBClassifier`.
