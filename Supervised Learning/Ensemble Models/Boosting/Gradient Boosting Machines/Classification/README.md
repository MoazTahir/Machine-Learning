<!-- markdownlint-disable MD013 -->
# Gradient Boosting — Wine Classification

This module packages a gradient boosting classifier for the classic UCI wine dataset. It follows the repository’s production blueprint: configuration-first design, reproducible training, persisted artefacts, FastAPI integration, and a companion notebook. Use it to explain boosting intuition, evaluate staged learning curves, and compare against bagging-based ensembles.

---

## Learning Objectives

- Revisit gradient boosting as stage-wise additive modelling that minimises empirical loss.
- Prepare and cache the scikit-learn wine dataset with deterministic preprocessing.
- Train, evaluate, and persist a `GradientBoostingClassifier` with tuned learning rate and depth.
- Serve predictions via FastAPI with calibrated class probabilities and surfaced feature importances.
- Extend the baseline with learning-rate schedules, early stopping, or monitoring integrations.

---

## Quickstart

1. **Install shared requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and persist artefacts**

   ```bash
   python "Supervised Learning/Ensemble Models/Boosting/Gradient Boosting Machines/Classification/src/train.py"
   ```

   Sample metrics:

   ```json
   {
     "accuracy": 0.978,
     "macro_precision": 0.979,
     "macro_recall": 0.975,
     "macro_f1": 0.977,
     "log_loss": 0.31,
     "roc_auc": 0.998
   }
   ```

3. **Run the shared API**

   ```bash
   python -m fastapi_app.main
   ```

4. **POST a prediction** (endpoint slug: `gradient_boosting_classification`):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/gradient_boosting_classification" \
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
       "class_0": 0.91,
       "class_1": 0.05,
       "class_2": 0.04
     },
     "model_version": "1731196800",
     "metrics": {
       "accuracy": 0.978,
       "macro_precision": 0.979,
       "macro_recall": 0.975,
       "macro_f1": 0.977,
       "log_loss": 0.31,
       "roc_auc": 0.998
     },
     "feature_importances": {
       "color_intensity": 0.17,
       "proline": 0.15,
       "flavanoids": 0.12,
       "alcohol": 0.10,
       "od280_od315_of_diluted_wines": 0.09,
       "...": "..."
     }
   }
   ```

5. **Explore the notebook**

   Open `notebooks/gradient_boosting_classification.ipynb` to inspect partial dependence plots, staged loss curves, and to mirror the scripted training routine in an interactive setting.

---

## Mathematical Foundations

Gradient boosting fits an ensemble of weak learners (shallow trees) in a stage-wise fashion. Each tree \(h_m(x)\) is trained to approximate the negative gradient of the loss function with respect to the model’s current predictions. For multi-class classification with log-loss, the additive model evolves as:

$$
F_0(x) = \arg\min_c \sum_i L(y_i, c), \qquad F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x),
$$

where \(\nu\) is the learning rate and \(h_m\) is fit to the residuals (negative gradients). Shrinking \(\nu\) and increasing the number of stages yields smoother fits that generalise better than single deep trees.

### Plain-Language Intuition

Think of a sommelier who repeatedly tastes wines and refines their scoring rules. The first round sets a rough baseline; subsequent rounds focus on bottles that were misclassified, adjusting the rules to correct previous mistakes. Learning rate controls how aggressively the sommelier updates their opinions, preventing overreactions to noisy samples.

---

## Dataset

- **Source**: UCI Wine dataset (178 samples, 13 continuous features across 3 cultivars).
- **Caching**: `data/wine.csv` generated on first run for reproducibility.
- **Target**: `class_label` with values `class_0`, `class_1`, `class_2`.

| Feature              | Description                                |
|----------------------|--------------------------------------------|
| `alcohol`            | Alcohol content                            |
| `color_intensity`    | Intensity of wine colour                   |
| `flavanoids`         | Phenolic compounds influencing taste       |
| `od280_od315...`     | Optical density metric                     |
| `proline`            | Amino acid correlated with body and aroma  |

Stratified 80/20 splits maintain class balance while the model retains a random seed for determinism.

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
│   └── gradient_boosting_classification.ipynb
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
- `data.py` fetches/normalises the dataset, persists a cached CSV, and serves stratified splits.
- `pipeline.py` strings together scaling and `GradientBoostingClassifier`, logs metrics, and dumps feature importances.
- `inference.py` exposes typed request/response models, lazy loading, and probability-aware predictions for FastAPI.
- `demo.py` offers a quick CLI smoke test without deploying the API layer.

---

## Extension Ideas

1. **Learning-rate schedules** — incorporate staged prediction monitoring (`staged_predict_proba`) to implement early stopping.
2. **Explainability** — add SHAP or partial dependence plots to the notebook and optionally surface them via the API.
3. **Hyperparameter sweeps** — random search over `learning_rate`, `n_estimators`, and `max_depth` to trade accuracy vs. inference time.
4. **Class-weighting** — experiment with class-specific weights if porting to imbalanced datasets.
5. **Pipeline hooks** — log training loss per iteration to your preferred experiment tracker (e.g., MLflow).

---

## References

- Friedman, Jerome H. "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 2001.
- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning.*
- scikit-learn documentation: `sklearn.ensemble.GradientBoostingClassifier`.
