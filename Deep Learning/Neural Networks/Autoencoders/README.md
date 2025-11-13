````markdown
# Autoencoders Roadmap

Comprehensive suite of Fashion-MNIST autoencoders implemented in both PyTorch and TensorFlow. Each sub-directory is self-contained with modular source code, documentation, and notebooks that follow a common workflow: configure → train → reconstruct → experiment.

| Variant | Motivation | Notes |
| ------- | ---------- | ----- |
| `Vanilla Autoencoder/` | Baseline reconstruction objective | Complete PyTorch + TensorFlow stacks |
| `Denoising Autoencoder/` | Robustness to Gaussian corruption | Noise injected in the dataloaders / `tf.data` pipeline |
| `Sparse Autoencoder/` | Encourage sparse latent activations | KL sparsity penalty with monitoring metrics |
| `Contractive Autoencoder/` | Penalise encoder sensitivity | Analytic Jacobian penalty for robustness |
| `Variational Autoencoder/` | Probabilistic latent space | Sampling helpers + KL tracking |

### How to use this section

1. Start with the vanilla module to familiarise yourself with the shared package layout.
2. Progress through denoising, sparse, contractive, and variational variants to explore increasingly advanced objectives.
3. Each module offers both PyTorch (`pytorch/`) and TensorFlow (`tensorflow/`) implementations, mirroring docs and notebooks for easy cross-framework comparisons.
4. Artefacts are saved under `artifacts/<framework>_<variant>/`, making it simple to compare metrics or reconstructions across experiments.

Looking for next steps? Extend these templates to convolutional autoencoders, add attention blocks, or integrate the training utilities into your own datasets by editing the config/data modules.

````