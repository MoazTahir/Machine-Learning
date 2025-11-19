# Machine-Learning

A comprehensive playground for classical machine learning, deep learning, and MLOps workflows. Each domain ships with curated datasets, notebooks, production-style `src/` packages, container-ready services, and documentation intended to double as learning material and implementation reference.

> [!NOTE]
> This repository evolves continuously. The roadmap and progress tables below reflect the latest completed and in-flight workstreams across supervised, unsupervised, deep learning, and operations tracks.

---

## Repository highlights

- **End-to-end verticals**: Every algorithm family includes a notebook for exploration, a `src/` module for reuse, artifacts for inference, and (when applicable) FastAPI/Gradio endpoints or Docker recipes.
- **Framework parity**: Deep-learning tracks (autoencoders, diffusion, GANs) provide mirrored PyTorch and TensorFlow implementations with matching configs, data pipelines, training engines, and inference helpers.
- **Documentation-first**: Each folder owns a README that explains the theory, architecture choices, experiment workflow, and troubleshooting tips for that scope.
- **Artifacts as first-class citizens**: Metrics, checkpoints, and sample grids persist under dedicated `artifacts/` directories, making it easy to compare runs or resume experiments.

---

## Repository structure

- `.dockerignore` — Docker build context filters used by containerised services.
- `.git/` — Git metadata (do not modify manually).
- `.gitignore` — Version-control ignore rules shared across all modules.
- `Deep Learning/` — Framework-specific learning paths (PyTorch/TensorFlow basics, neural-network subtracks, neural architecture search).
  - `Neural Networks/` — Detailed tracks for autoencoders, diffusion models, GANs, transformers, GNNs, RNNs, continual/meta learning, normalizing flows, and more.
- `Essentials Toolkit/` — Shared metric implementations, benchmark harness scaffolds, evaluation templates, and monitoring playbooks.
- `Evaluation/` — Operational checklists and forthcoming automation for experiment review.
- `fastapi_app/` — Unified FastAPI surface exposing trained models with Docker-ready deployment scripts.
- `LICENSE` — MIT license covering the repository.
- `Monitoring/` — Observability runbooks, logging templates, and future alerting integrations.
- `Reinforcement Learning/` — Planning/control curricula under construction with shared utilities and environment stubs.
- `Supervised Learning/` — Production-ready classical ML suites with datasets, notebooks, `src/` packages, artifacts, and service layers.
- `Unsupervised Learning/` — Clustering, dimensionality reduction, anomaly detection, and time-series analysis scaffolding with mirrored documentation.
- `requirements.txt` — Python dependencies for top-level workflows and notebooks.
- `README.md` — You are here; roadmap, navigation, and contribution guidance.

---

---

## Getting around

- Start with the algorithm-level README inside any folder; it links to prerequisite theory, notebook walkthroughs, and CLI commands.
- Cheat sheets for major pillars:
  - `Supervised Learning/README.md`
  - `Unsupervised Learning/README.md`
  - `Deep Learning/Neural Networks/README.md`
- Core utilities live under `Essentials Toolkit/` (metrics, benchmark harnesses, evaluation/monitoring playbooks).
- Each workflow directory (`data/`, `src/`, `notebooks/`, `artifacts/`) publishes a contract README describing file expectations, naming conventions, and automation hooks.

---

## Roadmap snapshot

### Supervised Learning (production ready)

- [x] Linear, logistic, and polynomial regression suites
- [x] Naive Bayes
- [x] Support Vector Machines (classification + regression)
- [x] Decision Trees (classification + regression)
- [x] Ensemble methods (Random Forest, Gradient Boosting, XGBoost, AdaBoost)
- [x] K-Nearest Neighbours (classification + regression)
- [x] Time-series forecasting pack (ARIMA, SARIMA, Prophet, Holt–Winters)

### Operations & Tooling

- [ ] Benchmark harnesses *(in design)*
- [ ] Evaluation playbooks *(in design)*
- [ ] Monitoring & observability runbooks *(in design)*

### Unsupervised Learning (scaffolding complete)

- [ ] K-Means, DBSCAN, Gaussian Mixtures
- [ ] PCA, ICA, Autoencoders for dimensionality reduction
- [ ] Anomaly detection workflows
- [ ] Time-series analysis notebook set (autocorrelation, seasonality, trend)

### Reinforcement Learning (upcoming)

- [ ] Q-Learning, SARSA, DQN variants
- [ ] Policy gradient methods, Actor–Critic
- [ ] Monte Carlo Tree Search, DDPG

### Deep Learning

- [x] Autoencoder track (vanilla, denoising, sparse, contractive, variational) — PyTorch + TensorFlow parity
- [x] Diffusion Models track — DDPM baseline with mirrored PyTorch/TensorFlow stacks, lab notebooks, and artifacts
- [x] Generative Adversarial Networks — DCGAN-style PyTorch and TensorFlow implementations with training/inference modules and guided notebooks
- [ ] Feedforward, CNN, RNN, and residual architectures *(scaffolding underway)*
- [ ] Graph/Boltzmann/Hopfield networks *(planned)*
- [ ] Neural Architecture Search *(planned)*

---

## Recent highlights (Q4 2025)

- **Generative module expansion**: Completed diffusion-model and GAN learning paths with unified READMEs, experiment prompts, and artifact pipelines.
- **Notebook-driven labs**: Every deep-learning module now includes a guided lab notebook covering setup → train → analyse → sample, plus next-step prompts.
- **Documentation uplift**: Root-level and framework-specific READMEs now catalog theory-to-code mappings, troubleshooting guides, and experiment backlogs.
- **Artifact harmonisation**: `artifacts/<track>/` directories share consistent naming (`metrics.json`, `{model}.pt`, `{model}_samples.png`) for easier automation.

---

## Contribution guidelines

Contributions keep this ecosystem growing. Bug fixes, new modules, documentation improvements, and reproducible experiments are all welcome.

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/awesome-improvement`.
3. Commit changes with clear messages: `git commit -m "Describe your change"`.
4. Push the branch: `git push origin feature/awesome-improvement`.
5. Open a pull request and tag it with the relevant track (supervised, unsupervised, deep-learning, ops).

> [!TIP]
> Include notebook outputs, metrics JSON, and sample artifacts when proposing model changes so reviewers can validate behaviour quickly.

---

## License

Distributed under the MIT License. See `LICENSE` for full terms.

---

## Contact

Mohammad Moaz Tahir  
LinkedIn: [https://www.linkedin.com/in/moaz-tahir](https://www.linkedin.com/in/moaz-tahir)  
Email: [moaztahir.mt@gmail.com](mailto:moaztahir.mt@gmail.com)
