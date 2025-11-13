````markdown
# Denoising Autoencoder

Introduces robustness to corrupted inputs by training the model to reconstruct clean Fashion-MNIST images from noisy variants. Both PyTorch and TensorFlow implementations share the same high-level workflow established in the vanilla module.

- `pytorch/` — Torch implementation with modular training loops and notebook walkthrough.
- `tensorflow/` — Keras mirror featuring a `tf.data` pipeline that generates noisy/clean pairs on the fly.

Suggested warm-up: revisit the vanilla autoencoder results, then compare PSNR curves after increasing `noise_std` here.

````