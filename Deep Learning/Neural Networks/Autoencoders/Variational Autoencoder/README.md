````markdown
# Variational Autoencoder

Probabilistic autoencoder that pairs an encoder producing Gaussian parameters with a decoder capable of sampling new digits. Both PyTorch and TensorFlow implementations follow the same layout as the other autoencoder variants.

- `pytorch/` — Torch VAE with KL tracking and sampling helpers.
- `tensorflow/` — Keras VAE with custom training step and notebook tour.

Use the provided notebooks to compare KL annealing schedules or visualise interpolations in latent space.

````