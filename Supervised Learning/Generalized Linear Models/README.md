# Generalised Linear Models

**Location:** `Machine-Learning/Supervised Learning/Generalized Linear Models`

This suite introduces count-based regression techniques where Gaussian assumptions break down. Each module adheres to the repo’s standard structure, enabling quick swaps between GLMs and existing baselines.

## Modules

- `Poisson Regression/` — Predict event counts with a log link and Poisson likelihood (scikit-learn implementation).
- `Negative Binomial Regression/` — Handle over-dispersed counts via `statsmodels` GLM utilities.

Use these when dealing with traffic, sales, or incident counts where variance grows with the mean.
