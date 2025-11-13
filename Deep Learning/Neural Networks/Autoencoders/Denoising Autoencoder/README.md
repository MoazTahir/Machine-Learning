````markdown
# Denoising Autoencoder

Strengthens the vanilla baseline by learning to map noisy Fashion-MNIST images back to their clean counterparts. Use this module to explore how explicit corruption changes the objective and the behaviour of the latent space.

---

## Learning goals

- Diagnose how different noise distributions and magnitudes affect reconstruction quality.
- Compare model resilience across frameworks by examining PSNR and MSE curves.
- Understand where to inject corruption in a data pipeline without changing model code.

---

## Directory tour

- `pytorch/` — Torch package with an on-the-fly noisy dataset wrapper, denoising inference helper, and guided notebook.
- `tensorflow/` — Keras mirror that builds paired noisy/clean batches via `tf.data`, complete with PSNR metric logging.

---

## Suggested experiments

1. Re-run the vanilla autoencoder notebook, then train the denoising version and chart the PSNR improvement on corrupted inputs.
2. Swap the Gaussian noise for salt-and-pepper or masking noise to see which corruptions are easiest to remove.
3. Transfer the trained encoder into the sparse or contractive variants to study the impact of combined objectives.

````