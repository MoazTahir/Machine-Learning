# Machine-Learning

## About The Project

This project contains different handcrafted Machine Learning Models with variety of training datasets, each tailored with best suited models. Inside, you'll find a well-organized folder structure containing relevant notebooks, Flask apps with Docker, Gradio interfaces for implementation, and/or inference models(applied if applicable). All this is avalible for you to learn about basics of AI/ML development.

> [!NOTE]
> This project is still under developement and the progress so far is mentioned in the Roadmap section

## Useage

Refer to the README inside each algorithm folder for end-to-end guidance (theory, training steps, notebooks, FastAPI integration, and demo scripts).
For a quick refresher across every supervised model, start with `Supervised Learning/README.md`.

> [!NOTE]
> Every project has its own different purpose, framework/tool-stack. As such, not every single one will have one or more of implementations mentioned in ***About The Project*** section

## Roadmap

Supervised
- [x] Linear Regression
- [x] Logistic Regression
- [x] Naive Bayes
- [x] Support Vector Machine
    - [x] Classification (Breast Cancer)
    - [x] Regression (California Housing)
- [x] Decision Tree
    - [x] Classification (Iris)
    - [x] Regression (California Housing)
- [ ] Ensembles
    - [ ] Bagging
        - [ ] Random Forest
    - [ ] Boosting
        - [ ] Gradient Boosting Machines
        - [ ] Stochastic Gradient Boosting
        - [ ] AdaBoost
        - [ ] XGBoost
- [x] K Nearest Neighbours
    - [x] Classification
    - [x] Regression
- [x] Time Series Forecasting
    - [x] ARIMA
    - [x] SARIMA
    - [x] Prophet
    - [x] Exponential Smoothing (Holt-Winters)

UnSupervised
- [ ] K Means Clustering
- [ ] DBSCAN
- [ ] Gaussian Mixture
- [ ] PCA
- [ ] ICA
- [ ] Anomaly Detection
- [ ] Time Series Analysis
    - [ ] Autocorrelation Analysis
    - [ ] Seasonality Decomposition
    - [ ] Trend Analysis

Reinforcement Learning
- [ ] Q-Learning
- [ ] Deep Q Network
- [ ] Deep SARSA
- [ ] Policy Gradient Methods
- [ ] Monte Carlo Tree Search
- [ ] Deep Deterministic Policy Gradients

Deep Learning
- [ ] PyTorch Basics (Started)
- [ ] Tensorflow Basics (Started)
- [ ] Neural Networks
    - [ ] FeedForward Neural Networks
    - [ ] Convolutional Neural Networks
    - [ ] Deconvolutional Neural Networks
    - [ ] MultiLayer Perceptrons(Feed Forward)
    - [ ] Generative Adversarial Networks
    - [ ] MLP
    - [ ] RNN
        - [ ] GRU
        - [ ] LSTM
    - [ ] Residual Networks
    - [ ] GANs
    - [ ] Boltzmann Machines
    - [ ] Hopfield Machines
    - [ ] Graph Neural Networks
- [ ] Neural Architecture Search

### Recent Supervised Updates

- Unified FastAPI endpoints now expose linear regression, logistic regression, Naive Bayes, SVM (classification), and SVR modules via the `fastapi_app` project.
- Each completed module includes a production-style `src/` package, exploratory notebook, notebook-ready dataset, persisted artefacts, and a `demo.py` sampler for quick predictions.
- READMEs now pair formal mathematical derivations with plain-language explanations to support both technical and non-technical audiences.
- KNN classification (wine) and regression (diabetes) modules follow the same pattern, including FastAPI services and notebooks.
- Time series forecasting modules (ARIMA, SARIMA, Prophet, Exponential Smoothing) now mirror the supervised-learning template with datasets, pipelines, notebooks, and inference services ready for deployment.
- A new supervised-learning cheat sheet (`Supervised Learning/README.md`) captures key interview-ready takeaways for every completed algorithm.
- Decision tree classification (Iris) and regression (California housing) modules provide interpretable baselines with feature importances surfaced via FastAPI endpoints and notebooks.
- Support vector machine classification and regression modules now live under a unified directory with mirrored structures, notebooks, and API endpoints for both tasks.

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Mohammad Moaz Tahir

* Linkedin : [https://www.linkedin.com/in/moaz-tahir](https://www.linkedin.com/in/moaz-tahir)
* Mail: moaztahir.mt@gmail.com
