# Poisson Regression — Count Modelling

**Location:** `Machine-Learning/Supervised Learning/Generalized Linear Models/Poisson Regression`

This module predicts event counts (e.g., support tickets per day) using a log-link Poisson regression. It demonstrates how to handle non-negative integer targets where variance scales with the mean.

## Features

- Synthetic dataset capturing baseline events, exposure, and promotions.
- scikit-learn `PoissonRegressor` pipeline with feature scaling.
- Evaluation metrics tailored to count data (Poisson deviance, mean absolute error).
- FastAPI-compatible inference service to plug into production workflows.
