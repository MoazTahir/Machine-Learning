<!-- markdownlint-disable MD013 -->
# K-Nearest Neighbours — Wine Classification

This module delivers a production-style K-Nearest Neighbours (KNN) classifier trained on the classic UCI Wine dataset. It follows the same structure as the other supervised-learning packages in this repository: reproducible configuration, scripted training, persisted artefacts, FastAPI wiring, an exploratory notebook, and a terse demo script. Use it to revisit distance-based classification, benchmark probabilistic baselines, or extend the project portfolio with a multi-class example.

---

## Learning Objectives

- Refresh the KNN algorithm: distance metrics, neighbour weighting, and decision boundaries.
- Reproduce the scikit-learn wine classification dataset with consistent feature naming.
- Train, evaluate, and persist a `StandardScaler` + `KNeighborsClassifier` pipeline.
- Serve probabilistic predictions through the shared FastAPI registry with typed payloads.
- Extend the baseline with hyperparameter sweeps, feature scaling experiments, or monitoring hooks.

---

## Quickstart

1. **Install shared dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and persist the model**

   ```bash
   python "Supervised Learning/K Nearest Neighbours/Classification/src/train.py"
   ```

   Example metrics output:

   ```json
   {
     "accuracy": 0.98,
     "macro_precision": 0.98,
     "macro_recall": 0.98,
     "macro_f1": 0.98
   }
   ```

   The script caches `artifacts/knn_classifier.joblib` and `artifacts/metrics.json`. The dataset is fetched from scikit-learn and stored at `data/wine.csv` on the first run.

3. **Launch the FastAPI service**

   ```bash
   python -m fastapi_app.main
   ```

4. **Submit a prediction** (after adding the registry entry)

   ```bash
   curl -X POST "http://127.0.0.1:8000/models/knn_classification" \
        -H "Content-Type: application/json" \
        -d '{
              "alcohol": 13.05,
              "malic_acid": 1.73,
              "ash": 2.04,
              "alcalinity_of_ash": 12.4,
              "magnesium": 92,
              "total_phenols": 2.72,
              "flavanoids": 3.27,
              "nonflavanoid_phenols": 0.17,
              "proanthocyanins": 1.98,
              "color_intensity": 3.0,
              "hue": 1.05,
              "od280_od315_of_diluted_wines": 3.58,
              "proline": 520
            }'
   ```

   Sample response:

   ```json
   {
     "predicted_label": "class_0",
     "class_probabilities": {
       "class_0": 0.86,
       "class_1": 0.14,
       "class_2": 0.0
     },
     "model_version": "1731196800",
     "metrics": {
       "accuracy": 0.98,
       "macro_precision": 0.98,
       "macro_recall": 0.98,
       "macro_f1": 0.98
     }
   }
   ```

5. **Explore the companion notebook**

   Open `notebooks/knn_classification.ipynb` to inspect the dataset, compare validation metrics, and visualise neighbourhood intuition.

6. **Optional Docker workflow**

   ```bash
   docker build -f fastapi_app/Dockerfile -t ml-fastapi .
   docker run --rm -p 8000:8000 ml-fastapi
   ```

   The container bundles every supervised module. On the first classification request the service trains automatically if artefacts are missing.

---

## Mathematical Foundations

KNN is a non-parametric method that assigns a label based on the majority class among the $k$ closest labelled samples. With distance metric $d(\cdot, \cdot)$, neighbour set $\mathcal{N}_k(x)$, and optional weights $w_i$, the predicted posterior is

$$
\hat{P}(y = c \mid x) = \frac{\sum_{x_i \in \mathcal{N}_k(x)} w_i \cdot \mathbf{1}[y_i = c]}{\sum_{x_i \in \mathcal{N}_k(x)} w_i}.
$$

This implementation uses Minkowski distance (equivalent to Euclidean when $p = 2$) with distance-based weights, giving nearer neighbours stronger influence. Scaling features with `StandardScaler` keeps distance comparisons balanced across heterogeneous magnitudes.

### Plain-Language Intuition

To classify an unknown wine, KNN looks at the seven closest wines in the training set (after standardisation) and counts how many belong to each variety. Each neighbour votes, but nearby wines vote louder than distant ones. The variety with the biggest weighted score wins, and normalising the scores yields the reported class probabilities.

---

## Dataset

- **Source**: UCI Wine dataset via `sklearn.datasets.load_wine` (178 samples, 13 physicochemical features, 3 classes).
- **Cache**: `data/wine.csv` (auto-generated).
- **Target**: `class_label` with values `class_0`, `class_1`, `class_2`.

Feature overview:

| Column                           | Description                                      |
|----------------------------------|--------------------------------------------------|
| `alcohol`                        | Alcohol content (percent by volume)              |
| `malic_acid`                     | Malic acid concentration                         |
| `ash`                            | Ash content                                      |
| `alcalinity_of_ash`              | Alkalinity of ash                                |
| `magnesium`                      | Magnesium concentration                          |
| `total_phenols`                  | Total phenolic content                           |
| `flavanoids`                     | Flavonoid concentration                          |
| `nonflavanoid_phenols`           | Non-flavonoid phenols                            |
| `proanthocyanins`                | Proanthocyanin concentration                     |
| `color_intensity`                | Color intensity measurement                      |
| `hue`                            | Hue measurement                                  |
| `od280_od315_of_diluted_wines`   | OD280/OD315 ratio of diluted wines               |
| `proline`                        | Proline concentration                            |

---

## Repository Layout

```
Classification/
├── README.md
├── data/
│   └── wine.csv                  # Cached dataset (auto-generated)
├── artifacts/
│   └── .gitkeep
├── notebooks/
│   └── knn_classification.ipynb  # Exploratory notebook mirroring src/
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

- `config.py` centralises feature names, neighbour settings, and local paths.
- `data.py` downloads the wine dataset, normalises column names, and provides stratified splits.
- `pipeline.py` chains standardisation with `KNeighborsClassifier`, logs macro metrics, and persists artefacts.
- `train.py` exposes the CLI entry point used in the Quickstart.
- `inference.py` defines the Pydantic schemas, primes a cached service, and plugs into the FastAPI registry.
- `demo.py` offers a rapid command-line prediction workflow for experimentation.

---

## Extensions

1. **Neighbour search experiments** — sweep `n_neighbors`, `weights`, and `metric` across a validation grid.
2. **Dimensionality reduction** — add PCA before KNN to compare performance and runtime.
3. **Explainability** — surface the neighbour set and distances alongside predictions for richer observability.
4. **Monitoring** — log class probability entropy to detect low-confidence regions in production data.
5. **Batch inference** — reuse `WineClassificationService` to score CSV batches offline.

---

## References

- Thomas M. Cover, Peter E. Hart. "Nearest neighbor pattern classification." *IEEE Transactions on Information Theory*, 1967.
- Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. *An Introduction to Statistical Learning*.
- scikit-learn documentation: `sklearn.datasets.load_wine`, `sklearn.neighbors.KNeighborsClassifier`.
