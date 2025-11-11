# Negative Binomial Regression — Over-dispersed Counts

**Location:** `Machine-Learning/Supervised Learning/Generalized Linear Models/Negative Binomial Regression`

Use this module when count data exhibit variance larger than the mean, violating Poisson assumptions. It leverages `statsmodels` to fit a Negative Binomial GLM with a log link.

## Highlights

- Synthetic call-center dataset with over-dispersed ticket counts.
- Train/evaluate workflow built on pandas + statsmodels.
- Metrics include pseudo R² and mean absolute error.
- FastAPI-compatible inference wrapper for easy deployment.
