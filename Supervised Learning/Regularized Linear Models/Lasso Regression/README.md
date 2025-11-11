# Lasso Regression — Sparse Feature Selection

**Location:** `Machine-Learning/Supervised Learning/Regularized Linear Models/Lasso Regression`

Lasso regression enforces an L1 penalty that pushes uninformative coefficients to exactly zero, making it invaluable for feature selection and high-dimensional tabular data. This module reuses the repo’s end-to-end scaffolding so you can slot lasso experiments into the existing FastAPI and benchmarking pipelines.

## Quickstart

1. Install the shared dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the baseline model:
   ```bash
   python "Supervised Learning/Regularized Linear Models/Lasso Regression/src/train.py"
   ```
3. Inspect `artifacts/metrics.json` and the coefficient sparsity summary to gauge how aggressively the penalty is pruning features.

## Structure

- `data/` — Synthetic dataset highlighting redundant features (`lasso_regression.csv`).
- `src/` — Config, data loading, pipeline, training CLI, and inference service.
- `notebooks/` — Suggested experiments for lambda sweeps and stability selection.
- `artifacts/` — Persisted model weights, metrics, and model card metadata.
- `demo.py` — Fire-and-forget script for spot-checking predictions.

## Next Steps

- Extend `pipeline.py` with cross-validation via `LassoCV`.
- Log coefficient paths through the notebooks for interpretability.
- Compare sparsity vs. metrics relative to ridge and elastic net modules.
