# Evaluation Playbooks

Authoritative recipes for assessing model quality across every stage of this repository.

---

## Core Principles

- **Consistency** — evaluation code lives beside each algorithm under `src/` and must yield the same metrics in notebooks, CLI runs, and API calls.
- **Reproducibility** — metrics are computed on deterministic splits (`random_state` pinned) and written to `artifacts/metrics.json` for auditability.
- **Comparability** — shared helpers in `Essentials Toolkit/Errors/metrics.py` define metric formulas, so cross-project scorecards use identical implementations.
- **Actionability** — every evaluation report should point to a decision: ship, tune, or roll back.

## Standard Workflow

1. **Establish baselines**
	- Load the dataset with `src/data.py` helpers.
	- Fit a naive or historical baseline (e.g. mean predictor for regression, majority class for classification).

2. **Train candidate models**
	- Use the algorithm-specific pipeline (`src/pipeline.py`) and configuration dataclasses to ensure identical preprocessing at inference time.
	- Persist artefacts via `train_and_persist` so the FastAPI service and demo scripts stay aligned.

3. **Score and validate**
	- Compute the canonical metric bundle per task type:

	  | Task Type        | Required Metrics                                                           |
	  |------------------|----------------------------------------------------------------------------|
	  | Regression       | `r2`, `rmse`, `mae`, optional `mape`, `smape`, prediction interval width  |
	  | Classification   | `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, optional `pr_auc`     |
	  | Clustering       | `silhouette`, `davies_bouldin`, `calinski_harabasz`                       |
	  | Time Series      | `rmse`, `mae`, `mase`, seasonal naive delta, coverage of prediction bands |

	- For stochastic training, wrap scoring with cross-validation or repeated hold-out and log the mean/variance.

4. **Diagnose**
	- Generate residual plots, confusion matrices, feature importance charts, and calibration curves in notebooks or automated reports.
	- Compare metrics across key slices (e.g. geography, class imbalance) to surface fairness or drift risks early.

5. **Publish artefacts**
	- Write metrics to `artifacts/metrics.json` using the pipeline utilities (see `LinearRegressionPipeline.write_metrics`).
	- Emit a JSON schema-aligned summary so monitoring services can ingest the data without custom parsing.

## Automation Hooks

- **Experiment tracking** — plug the output dictionary from `train_and_persist` into MLflow, Weights & Biases, or lightweight CSV/Markdown logs.
- **Unit tests** — gate merges with assertions on minimum acceptable metrics (example: `assert metrics['r2'] > 0.9`).
- **Continuous evaluation** — schedule periodic re-training jobs that compare new metrics against the last signed-off run and notify when deviations exceed configured tolerances.

## Extending The Playbooks

- Add task-specific notebooks under `Evaluation/notebooks/` demonstrating end-to-end evaluation pipelines with commentary.
- Contribute reusable scoring scripts (e.g. `evaluation/regression_scorecard.py`) that hydrate the config objects and emit both tabular and visual outputs.
- Document statistical techniques such as paired t-tests, bootstrap confidence intervals, or sequential testing for online experiments.

As new models land in the repository, update this playbook with concrete examples and code excerpts so contributors can immediately reproduce the evaluation posture in their own modules.
