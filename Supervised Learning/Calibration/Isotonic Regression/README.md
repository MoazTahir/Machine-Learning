# Isotonic Regression — Non-parametric Calibration

**Location:** `Machine-Learning/Supervised Learning/Calibration/Isotonic Regression`

Isotonic regression fits a monotonic calibration curve to raw classifier scores, making it ideal when probabilities need flexible adjustment beyond a sigmoid.

## Highlights

- Same synthetic dataset as Platt scaling for baseline comparison.
- scikit-learn `CalibratedClassifierCV` with `method="isotonic"`.
- Calibration diagnostics reported via reliability diagrams and ECE.
- FastAPI-ready inference service returning calibrated probabilities.
