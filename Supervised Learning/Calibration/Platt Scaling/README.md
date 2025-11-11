# Platt Scaling — Sigmoid Calibration

**Location:** `Machine-Learning/Supervised Learning/Calibration/Platt Scaling`

Applies a logistic/sigmoid calibration layer to raw classifier scores, improving probability estimates for downstream decision making.

## Highlights

- Synthetic binary dataset with intentional class imbalance.
- Baseline logistic regression followed by scikit-learn `CalibratedClassifierCV` (`method="sigmoid"`).
- Metrics include Brier score loss and expected calibration error (ECE).
- FastAPI service exposing calibrated probabilities for integration tests.
