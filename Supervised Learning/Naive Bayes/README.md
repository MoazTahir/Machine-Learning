<!-- markdownlint-disable MD013 -->
# Naive Bayes — Mushroom Edibility Classification

This module packages an end-to-end workflow for categorical Naive Bayes on the classic mushroom dataset. It blends the statistical perspective, a reproducible scikit-learn pipeline, notebook-driven exploration, and a FastAPI-ready inference service. Use it as a template for other categorical problems, a teaching aid on probabilistic classifiers, or a springboard for production deployments.

---

## Learning Objectives

- Understand the multinomial (categorical) Naive Bayes model and the role of conditional independence.
- Explore the mushroom dataset, handle missing categorical values, and engineer consistent encodings.
- Train, evaluate, and persist a CategoricalNB pipeline with one-hot features.
- Serve edibility predictions through the shared FastAPI registry with structured request/response contracts.
- Extend the pipeline with feature grouping, prior tuning, or alternative probabilistic models.

---

## Quickstart

1. **Install the shared environment**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model and persist artefacts**

   ```bash
   python "Supervised Learning/Naive Bayes/src/train.py"
   ```

   Example output:

   ```json
   {
     "accuracy": 0.973,
     "precision": 0.970,
     "recall": 0.977,
     "f1": 0.974,
     "roc_auc": 0.996
   }
   ```

   The script writes `artifacts/naive_bayes_model.joblib` and `artifacts/metrics.json`.

3. **Launch the unified inference API**

   ```bash
   python -m fastapi_app.main
   ```

4. **Issue a prediction** (interactive docs are available at `http://127.0.0.1:8000/docs` once the model is registered in `fastapi_app/services/registry.py`):

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/naive_bayes" \
        -H "Content-Type: application/json" \
        -d '{
              "cap-shape": "x",
              "cap-surface": "s",
              "cap-color": "n",
              "bruises": "t",
              "odor": "a",
              "gill-attachment": "f",
              "gill-spacing": "c",
              "gill-size": "b",
              "gill-color": "k",
              "stalk-shape": "e",
              "stalk-root": "b",
              "stalk-surface-above-ring": "s",
              "stalk-surface-below-ring": "s",
              "stalk-color-above-ring": "w",
              "stalk-color-below-ring": "w",
              "veil-type": "p",
              "veil-color": "w",
              "ring-number": "o",
              "ring-type": "p",
              "spore-print-color": "k",
              "population": "s",
              "habitat": "u"
            }'
   ```

   Sample response:

   ```json
   {
     "predicted_label": "poisonous",
     "probability_poisonous": 0.91,
     "model_version": "1730764800",
     "metrics": {
       "accuracy": 0.973,
       "precision": 0.970,
       "recall": 0.977,
       "f1": 0.974,
       "roc_auc": 0.996
     }
   }
   ```

5. **Explore the companion notebook**

   Open `notebooks/naive_bayes.ipynb` to reproduce the pipeline step-by-step, inspect conditional probabilities, and visualise decision boundaries in the encoded feature space.

6. **Run the API in Docker** (optional containerised workflow)

   ```bash
   docker build -f fastapi_app/Dockerfile -t ml-fastapi .
   docker run --rm -p 8000:8000 ml-fastapi
   ```

   The container includes all supervised-learning modules. On the first request the service trains Naive Bayes automatically if `artifacts/` is empty.

---

## Mathematical Foundations

Categorical Naive Bayes models the posterior class probability by assuming conditional independence of features given the class label:

$$
P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y).
$$

For mushroom edibility, each feature (cap shape, odor, habitat, …) can take a finite set of symbols. The classifier learns class-conditional likelihood tables for every feature value using maximum likelihood with Laplace smoothing:

$$
P(x_i = v \mid y) = \frac{N_{y,v} + \alpha}{N_y + \alpha |\mathcal{V}_i|},
$$

where $N_{y,v}$ counts how often value $v$ occurs among class $y$, $|\mathcal{V}_i|$ is the cardinality of feature $i$, and $\alpha$ is the smoothing strength (defaults to 1 in `CategoricalNB`). Taking the logarithm converts the product into a sum, which is numerically stable and computationally cheap. Despite the bold independence assumption, Naive Bayes performs exceptionally well on this dataset because many features provide strong, near-orthogonal signals about toxicity.

### Assumptions to monitor

1. **Conditional independence** — correlated categorical features reduce calibration quality; watch for highly redundant encodings.
2. **Complete coverage** — unseen categories trigger the `OneHotEncoder`/`CategoricalNB` handling pipeline; verify metrics if you prune rare values.
3. **Stationary distributions** — the prior and likelihood tables assume data is i.i.d.; monitor drift if you deploy the service with new mushroom variants.

---

### Plain-Language Intuition

Naive Bayes is essentially a counting machine. It scans the training mushrooms, records how often each smell, cap colour, and habitat appears for the edible vs. poisonous group, and turns those tallies into simple percentages. When you pass in a new mushroom, the model grabs the relevant percentages for each trait and multiplies them together, producing a score for “poisonous” and another for “edible.” Whichever total is larger wins, and the ratio between them becomes the reported probability.

The “naive” part comes from pretending the traits are independent—even though, in reality, they might influence one another. Surprisingly, as long as each trait gives a strong hint about safety on its own (odor is a great example), combining these hints still works very well, which is why the model nails this dataset despite the simplifying assumption.

## Dataset

- **Source**: UCI Mushroom dataset (8,124 mushrooms, 22 categorical attributes, binary edibility label).
- **Location**: `data/mushrooms.csv`.
- **Target encoding**: `class = 1` (poisonous) vs. `class = 0` (edible) in the persisted features.
- **Missing values**: denoted by `?` in the raw CSV; replaced with `pd.NA` then imputed with the modal value during preprocessing.

Feature snapshot:

| Column                    | Description                                              |
|---------------------------|----------------------------------------------------------|
| `cap-shape`               | {bell, conical, convex, flat, knobbed, sunken}           |
| `odor`                    | {almond, anise, creosote, fishy, foul, musty, none, pungent, spicy} |
| `gill-color`              | 12 hue categories describing gill pigmentation           |
| `ring-type`               | {evanescent, flaring, large, none, pendant}              |
| `habitat`                 | {grasses, leaves, meadows, paths, urban, waste, woods}   |

The training routine performs an 80/20 stratified split so class balance remains consistent across train and validation folds.

---

## Repository Layout

```
Naive Bayes/
├── README.md
├── data/
│   └── mushrooms.csv
├── notebooks/
│   └── naive_bayes.ipynb
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

The structure mirrors the other supervised-learning modules so you can diff approaches quickly.

---

## Implementation Walkthrough

### Code modules (`src/`)

- `config.py` defines `NaiveBayesConfig`, centralising file paths, feature names, and train/test parameters while ensuring artefact directories exist.
- `data.py` loads the CSV, normalises missing markers, constructs `(X, y)`, and performs stratified splits.
- `pipeline.py` builds a preprocessing stack (categorical imputation + one-hot encoding) feeding `CategoricalNB`, trains the model, computes accuracy/precision/recall/F1/ROC-AUC, and persists artefacts via joblib/JSON.
- `train.py` acts as the CLI entry point, calling `train_and_persist` and printing the metrics dictionary for logging.
- `inference.py` specifies the Pydantic request/response schemas, caches a `NaiveBayesService`, lazily trains if artefacts are missing, and exposes `RequestModel`, `ResponseModel`, and `get_service()` for FastAPI integration.

### Serving via FastAPI

Once you declare the slug inside `fastapi_app/services/registry.py`, the shared API layer automatically:

- Validates incoming JSON against `NaiveBayesRequest` (hyphenated feature names included).
- Loads cached artefacts (training on demand if necessary).
- Returns edibility predictions alongside the persisted metrics and a timestamp-derived model version.

Leverage the built-in `/models/{slug}` routes for consistent monitoring and logging across all algorithms.

---

## Notebook Tour

`notebooks/naive_bayes.ipynb` echoes the production pipeline and adds rich context:

1. Dataset exploration with frequency tables and mutual information plots.
2. Training walkthrough mirroring `src/pipeline.py`, including confusion matrix and ROC curves.
3. Feature importance proxies via likelihood ratios to explain why certain odors dominate the decision.
4. Experiments with smoothing strength (`alpha`) and grouped feature encodings to test robustness.
5. Guidance on exporting notebook insights back into the FastAPI service or alternative deployment targets.

Running the notebook after changes ensures the scripted pipeline and exploratory workflow stay aligned.

---

## Extending the Module

1. **Group rare categories** — collapse low-frequency values (e.g., habitats) to improve calibration or reduce dimensionality.
2. **Prior tuning** — adjust class priors in `CategoricalNB` if your deployment distribution differs from training data.
3. **Explainability** — log the top-$k$ likelihood ratios per request to highlight the strongest poisonous signals.
4. **Batch scoring** — repurpose `NaiveBayesService` in a CLI that reads CSV batches and streams predictions to cloud storage.
5. **Model comparison** — plug complementary algorithms (Random Forest, Gradient Boosting) into the same artefact layout for A/B testing.

---

## References

- Harry Zhang. "The Optimality of Naive Bayes." *AAAI*, 2004.
- Kevin Patrick Murphy. *Machine Learning: A Probabilistic Perspective.*
- UCI Machine Learning Repository — Mushroom dataset documentation.

Consult these resources alongside the notebook to deepen intuition and adapt the pipeline to other categorical domains.
