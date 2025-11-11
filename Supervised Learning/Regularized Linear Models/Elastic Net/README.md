# Elastic Net Regression — Hybrid Regularisation

**Location:** `Machine-Learning/Supervised Learning/Regularized Linear Models/Elastic Net`

Elastic net blends L1 and L2 penalties, offering the sparsity of lasso with the stability of ridge. Use it when correlated feature groups exist and you still want automatic variable selection.

## Highlights

- Shared scaffolding with the rest of the repository (data, src, notebooks, artifacts, demo).
- Default configuration leans toward ridge (`l1_ratio=0.5`); tweak the blend to emphasise sparsity vs. shrinkage.
- Metrics track both standard regression scores and the proportion of coefficients shrunk to zero.

## Getting Started

```bash
pip install -r requirements.txt
python "Supervised Learning/Regularized Linear Models/Elastic Net/src/train.py"
```

Inspect the resulting `artifacts/metrics.json` and coefficient CSV to understand how different `alpha` and `l1_ratio` choices affect the solution.
