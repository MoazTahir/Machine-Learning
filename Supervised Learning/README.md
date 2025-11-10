# Supervised Learning Cheat Sheet

Fast refresher on the supervised models hosted in this workspace. Skim the bullets before an interview, then dive into each module's README for code-level detail.

## Linear Regression
- **Core idea:** Fit a straight-line relationship between features and a continuous target by minimising mean squared error.
- **Use it when:** Relationships look roughly linear and interpretability matters; baseline every regression task with it.
- **Dial in:** Standardise features, check multicollinearity, and add regularisation (ridge/lasso) if coefficients explode.
- **Watch out:** Outliers and non-stationary data can dominate the fit—inspect residual plots and consider transformations.

## Logistic Regression
- **Core idea:** Estimate class probabilities via the sigmoid of a linear combination of features; interpret coefficients as log-odds.
- **Use it when:** You need a quick, explainable classifier for linearly separable problems or as a calibration baseline.
- **Dial in:** Scale inputs, balance classes (weights or sampling), and explore regularisation strength (`C` in scikit-learn).
- **Watch out:** High-dimensional sparse data can overfit without penalty terms—keep an eye on validation loss.

## Naive Bayes
- **Core idea:** Apply Bayes' theorem assuming feature independence, trading realism for speed and robustness on small data.
- **Use it when:** Working with text (bag-of-words), categorical features, or needing a surprisingly strong baseline.
- **Dial in:** Choose the right variant (Gaussian, Multinomial, Bernoulli) to match feature distributions.
- **Watch out:** Correlated features break the independence assumption; consider feature hashing or selection first.

## Support Vector Machines
### Classification (Breast Cancer)
- **Core idea:** Find the maximum-margin hyperplane separating classes, optionally projecting data into higher dimensions with kernels.
- **Use it when:** You have medium-sized datasets with clear margins or overlapping classes that benefit from kernel tricks.
- **Dial in:** Grid-search `C` and `gamma`, pick kernels (`linear`, `rbf`, `poly`) that reflect decision boundary complexity.
- **Watch out:** Scaling is mandatory; SVMs do not handle noisy, overlapping classes without careful tuning.

### Regression (California Housing)
- **Core idea:** Extend SVM to regression by fitting within an epsilon-insensitive tube, penalising only large deviations.
- **Use it when:** You need robust regression that ignores small errors and captures non-linear trends with kernels.
- **Dial in:** Adjust `C`, `epsilon`, and kernel parameters to balance flatness vs. sensitivity.
- **Watch out:** Performance degrades on large datasets due to quadratic complexity; sample or move to linear SVR when needed.

## K Nearest Neighbours (KNN)
### Classification
- **Core idea:** Predict the class most common among the `k` closest training examples using a distance metric.
- **Use it when:** Decision boundaries are irregular and you have a modest dataset with meaningful local structure.
- **Dial in:** Experiment with `k`, distance metrics (Euclidean, Manhattan), and weighting neighbours by inverse distance.
- **Watch out:** Scaling is essential; KNN is sensitive to irrelevant features and becomes slow with large datasets.

### Regression
- **Core idea:** Average the targets of the `k` nearest neighbours to estimate a continuous value.
- **Use it when:** Local patterns matter more than a global function and interpretability is less critical.
- **Dial in:** Tune `k`, distance weighting, and feature scaling; cross-validate to avoid over-smoothing or noisy predictions.
- **Watch out:** Outliers and heterogeneous feature scales distort neighbourhoods—normalise and consider outlier clipping.

## Decision Tree
### Classification (Iris)
- **Core idea:** Grow axis-aligned splits that maximise class purity and report feature importances for transparency.
- **Use it when:** You want an interpretable baseline on small-to-medium tabular data or need probability outputs without heavy tuning.
- **Dial in:** Adjust `max_depth`, `min_samples_leaf`, and `ccp_alpha` to balance bias/variance; inspect feature importances for sanity checks.
- **Watch out:** Deep trees memorise noise—validate with stratified splits and prune or cap depth when metrics diverge.

### Regression (California Housing)
- **Core idea:** Segment the feature space into regions with similar target means, predicting the average value per leaf.
- **Use it when:** You need quick non-linear baselines with explainable splits and ranked feature importance.
- **Dial in:** Tune depth/leaf thresholds and cost-complexity pruning; bucket geographic features when necessary.
- **Watch out:** Piecewise-constant predictions can jump at split boundaries; monitor residuals and revisit settings if variance spikes.

## Ensemble Models *(scaffolding in place)*
### Bagging / Random Forest
- **Why it helps:** Aggregates many de-correlated trees to reduce variance while preserving low bias.
- **Remember:** Set enough estimators, use out-of-bag score for quick validation, and tune max features for diversity.
- **Interview tip:** Highlight robustness to missing data and natural feature importance metrics.

### Boosting (GBM, Stochastic GBM, AdaBoost, XGBoost)
- **Why it helps:** Sequentially focuses on residual errors, building strong learners from weak base models.
- **Remember:** Learning rate vs. number of estimators is the key trade-off; shrinkage plus shallow trees works well.
- **Interview tip:** Mention regularisation options (subsample, column sample) and that XGBoost handles sparsity efficiently.

## Time Series Forecasting
- **Shared setup:** All modules load the AirPassengers dataset, split chronologically, log MAE/RMSE/MAPE, and persist models with Joblib for FastAPI inference.

### ARIMA
- **Core idea:** Combine autoregression (AR), differencing (I), and moving averages (MA) to model stationary series.
- **Use it when:** After differencing removes trend/seasonality and residuals look white-noise.
- **Watch out:** Over-differencing kills signal; rely on ACF/PACF and diagnostics plots.

### SARIMA
- **Core idea:** Extend ARIMA with seasonal components to capture repeating yearly or monthly patterns.
- **Use it when:** Seasonality is obvious (e.g., monthly passengers) and needs explicit modelling.
- **Watch out:** Too many seasonal parameters explode training time—start small and inspect residuals.

### Prophet
- **Core idea:** Decompose time series into trend, seasonality, and holidays with an additive model that handles missing data gracefully.
- **Use it when:** You want fast, auto-tuned forecasts with interpretable components and built-in uncertainty intervals.
- **Watch out:** Default changepoint priors can underfit sudden regime shifts; loosen `changepoint_prior_scale` when trends jump.

### Exponential Smoothing (Holt-Winters)
- **Core idea:** Smooth level, trend, and seasonality exponentially, giving more weight to recent observations.
- **Use it when:** Seasonality is regular and you prefer a fast, classical baseline.
- **Watch out:** Incorrect seasonal period or damping choices lead to drift—double-check seasonal length and residual plots.

---
**How to use this file:** Skim before coding rounds, then open the specific module for hands-on pipelines, notebooks, and FastAPI services.
