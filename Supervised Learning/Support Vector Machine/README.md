<!-- markdownlint-disable MD013 -->
# Support Vector Machine Modules

This directory hosts the support vector family of models, grouped into dedicated classification and regression submodules. Each submodule mirrors the repository-wide structure with reproducible training scripts, persisted artefacts, exploratory notebooks, and FastAPI-ready inference services.

## Submodules

- `Classification/` — Breast cancer diagnosis using an RBF-kernel SVC with probability calibration and feature scaling.
- `Regression/` — California housing price prediction via RBF-kernel SVR with quantile-aware evaluation and feature importances.

Refer to the README inside each submodule for datasets, CLI commands, notebooks, and extension ideas.
