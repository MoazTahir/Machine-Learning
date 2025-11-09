<!-- markdownlint-disable MD013 -->
# Logistic Regression — Heart Disease Classification

This module delivers a complete learning and production toolkit for binary logistic regression. You will find in-depth theory, a reproducible training pipeline, an exploratory notebook, and a pluggable FastAPI service that exposes the trained classifier. Use it as a refresher on medical risk modelling, a teaching aid, or a template for other classification problems.

---

## Quickstart

1. **Install the shared environment**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model and persist artefacts**

   ```bash
   python "Supervised Learning/Logistic Regression/src/train.py"
   ```

   Example output:

   ```json
   {
     "accuracy": 0.885,
     "precision": 0.893,
     "recall": 0.862,
     "f1": 0.877,
     "roc_auc": 0.941
   }
   ```

   The script writes `artifacts/logistic_regression_model.joblib` and `artifacts/metrics.json`.

3. **Launch the unified inference API**

   ```bash
   python -m fastapi_app.main
   ```

4. **Issue a prediction** (FastAPI docs at `http://127.0.0.1:8000/docs` provide an interactive form):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/logistic_regression" \
        -H "Content-Type: application/json" \
        -d '{
              "age": 54,
              "sex": 1,
              "cp": 1,
              "trestbps": 130,
              "chol": 246,
              "fbs": 0,
              "restecg": 1,
              "thalach": 150,
              "exang": 0,
              "oldpeak": 1.0,
              "slope": 1,
              "ca": 0,
              "thal": 2
            }'
   ```

   Sample response:

   ```json
   {
     "predicted_class": 0,
     "probability": 0.23,
     "model_version": "1730764800",
     "metrics": {
       "accuracy": 0.885,
       "precision": 0.893,
       "recall": 0.862,
       "f1": 0.877,
       "roc_auc": 0.941
     }
   }
   ```

5. **Explore the companion notebook**

   Open `notebooks/logistic_regression.ipynb` to walk through the data story, feature engineering ideas, ROC analysis, and calibration experiments.

6. **Run the API in Docker** (optional containerised workflow)

   ```bash
   docker build -f fastapi_app/Dockerfile -t ml-fastapi .
   docker run --rm -p 8000:8000 ml-fastapi
   ```

---

## Mathematical Foundations

Binary logistic regression models log-odds of the positive class as a linear combination of predictors:

$$
\log \frac{P(y=1 \mid x)}{1 - P(y=1 \mid x)} = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p.
$$

Solving for the probability yields the familiar sigmoid:

$$
P(y=1 \mid x) = \sigma(\beta_0 + \beta^\top x) = \frac{1}{1 + e^{-(\beta_0 + \beta^\top x)}}.
$$

### Interpreting coefficients

- Positive coefficients increase the log-odds of heart disease. For instance, a positive weight on `cp` (chest pain type) indicates that higher values correlate with higher risk.
- Exponentiating coefficients (`e^{\beta_i}`) yields the odds ratio — “how many times more likely” the outcome becomes per unit increase.

### Assumptions worth validating

1. **Linearity in the logit** — the relationship between predictors and log-odds should be linear. We include polynomial interaction analysis in the notebook to probe deviations.
2. **Independence of observations** — satisfied for patient-level records collected independently.
3. **Low multicollinearity** — high correlation between features inflates variance; check VIF or correlation heatmaps.
4. **Well-calibrated probabilities** — ROC and calibration curves in the notebook confirm the classifier’s probabilistic quality.

---

### Plain-Language Intuition

You can think of logistic regression as a weighted checklist for heart health. Each feature (age, blood pressure, cholesterol, etc.) gets a weight that either pushes the score toward “disease” or “no disease.” After adding up the weighted checklist, the model runs the total through an S-shaped curve that squeezes any number into the 0–1 range, giving you a probability instead of a raw score. Values near 1 mean the checklist items pointed strongly toward disease; values near 0 mean the evidence leaned the other way.

During training the model looks at many past patients, nudging the weights so that the predicted probabilities line up with the observed outcomes. The evaluation metrics summarise how often those probabilities translate into correct decisions and how confident the model can be when distinguishing risky patients from healthy ones.

## Dataset

- **Source**: UCI Heart Disease dataset.
- **Location**: `data/heart.csv`.
- **Prediction target**: `target = 1` indicates presence of heart disease.
- **Feature snapshot**:

  | Feature   | Description                                 |
  |-----------|---------------------------------------------|
  | `age`     | Age in years                                 |
  | `sex`     | 1 = male, 0 = female                         |
  | `cp`      | Chest pain type (0–3)                        |
  | `trestbps`| Resting blood pressure (mm Hg)               |
  | `chol`    | Serum cholesterol (mg/dl)                    |
  | `fbs`     | Fasting blood sugar > 120 mg/dl (1/0)        |
  | `restecg` | Resting ECG results (0–2)                    |
  | `thalach` | Maximum heart rate achieved                  |
  | `exang`   | Exercise-induced angina (1/0)                |
  | `oldpeak` | ST depression induced by exercise            |
  | `slope`   | Slope of peak exercise ST segment (0–2)      |
  | `ca`      | Number of major vessels coloured (0–4)       |
  | `thal`    | Thalassemia category (3, 6, 7)               |

The training script performs an 80/20 stratified split so class proportions remain stable across train and validation folds.

---

## Repository Layout

```
Logistic Regression/
├── README.md
├── data/
│   └── heart.csv
├── notebooks/
│   └── logistic_regression.ipynb
├── src/
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

### Training pipeline (`src/`)

- `config.py` centralises feature names, file paths, stratification size, and ensures artefact directories exist.
- `data.py` loads the CSV, constructs `(X, y)`, and performs stratified splits via `train_test_split`.
- `pipeline.py` wraps a `StandardScaler` + `LogisticRegression` inside a scikit-learn `Pipeline`, trains it, evaluates accuracy/precision/recall/F1/ROC-AUC, and persists both model weights (`joblib`) and metrics (`json`).
- `train.py` is the CLI entrypoint invoked in the Quickstart. It prints the metrics dictionary for easy logging.
- `inference.py` defines Pydantic request/response models plus a cached service (`get_service()`) used by the shared FastAPI registry.

### Serving via FastAPI

The FastAPI project (`fastapi_app/`) loads this module dynamically:

- `services/registry.py` registers the `logistic_regression` slug and lazily imports `src/inference.py`.
- When a request hits `/models/logistic_regression`, the registry instantiates `LogisticRegressionService`, loads artefacts (training on the fly if needed), validates inputs with Pydantic, and returns probabilities alongside saved metrics.
- Metrics attached to responses make it easy to monitor for drift or stale models in downstream clients.

---

## Notebook Tour

`notebooks/logistic_regression.ipynb` contains:

1. Statistical refresher on the logit link and odds ratios.
2. Exploratory plots (pairplots, heatmaps) to understand feature interactions.
3. Training walkthrough mirroring `src/pipeline.py` with confusion matrix visualisations.
4. ROC curve, precision–recall curve, and calibration plot to assess probabilistic performance.
5. Guidance on exporting notebook insights back into the production code.

Executing the notebook after modifying `src/` code is a quick regression test — metric parity should match the output of `train.py`.

---

## Extending the Module

1. **Feature engineering** — introduce interaction terms or polynomial features by editing `_build_preprocessing()` in `pipeline.py`.
2. **Threshold tuning** — adjust the decision threshold (currently 0.5) inside `LogisticRegressionService.predict` based on ROC analysis.
3. **Regularisation sweep** — replace `LogisticRegression` with the `saga` solver and add `penalty`/`C` hyperparameters for elastic-net search.
4. **Explainability** — integrate SHAP or LIME in the notebook to surface patient-level explanations.
5. **Monitoring** — extend the FastAPI response payload with timestamped metrics or integrate Prometheus counters in the service layer.

---

## References

- David W. Hosmer Jr., Stanley Lemeshow, Rodney X. Sturdivant. *Applied Logistic Regression.*
- Max Kuhn, Kjell Johnson. *Applied Predictive Modeling.*
- UCI Machine Learning Repository — Heart Disease dataset.

Use these resources alongside the notebook to deepen understanding and to adapt the module for real-world clinical risk scoring pipelines.
