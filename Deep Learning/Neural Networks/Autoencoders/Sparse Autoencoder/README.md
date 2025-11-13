````markdown
# Sparse Autoencoder

Applies a KL divergence penalty to enforce sparse activations in the latent space. The directory mirrors the structure established in the vanilla and denoising modules with PyTorch and TensorFlow implementations.

- `pytorch/` — Torch module with explicit KL penalties inside the training loop.
- `tensorflow/` — Keras custom training loop achieving the same objective.

Experiment idea: compare reconstruction PSNR across the vanilla, sparse, and contractive variants using the provided metrics logs.

````