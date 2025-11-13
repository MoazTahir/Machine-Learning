````markdown
# Contractive Autoencoder

Promotes robustness to local perturbations by penalising the sensitivity of the encoder. The directory mirrors prior modules with PyTorch and TensorFlow implementations.

- `pytorch/` — Torch module that adds a Jacobian-based penalty inside the training loop.
- `tensorflow/` — Keras model with a custom training step implementing the same penalty.

Try comparing latent traversals between the vanilla and contractive variants to see how the penalty shapes the embedding geometry.

````