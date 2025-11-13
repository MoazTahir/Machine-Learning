````markdown
# Sparse Autoencoder

Targets compact latent representations by adding a KL divergence penalty that pushes activations toward a low desired firing rate.

---

## Learning goals

- See how sparsity regularisation influences encoder behaviour and reconstruction quality.
- Compare how PyTorch and TensorFlow implement custom penalties (loss augmentation vs overridden `train_step`).
- Use logged KL metrics to diagnose under- or over-regularisation.

---

## Directory tour

- `pytorch/` — Torch pipeline with KL divergence utilities and a metrics-rich training loop.
- `tensorflow/` — Subclassed Keras model that integrates the penalty directly inside the optimisation step.

---

## Suggested experiments

1. Plot latent activation histograms across epochs to confirm the sparsity target is met.
2. Combine the sparsity penalty with denoising by importing the noisy dataset wrapper from the previous module.
3. Vary latent dimensionality to examine whether the penalty or bottleneck dominates the learning dynamics.

````