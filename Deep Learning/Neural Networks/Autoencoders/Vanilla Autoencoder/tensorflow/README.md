# TensorFlow Vanilla Autoencoder

A Keras implementation of a fully-connected autoencoder trained on Fashion-MNIST.

---

## 1. Notebook tour

- `notebooks/vanilla_autoencoder_tensorflow.ipynb` mirrors the same three-step routine used in other modules: import `../src`, call `train.train()`, then rebuild a sample image.
- Metrics include MSE and a custom PSNR monitor implemented with `tf.image.psnr`.

---

## 2. Source layout

| File | Purpose |
| ---- | ------- |
| `config.py` | Paths, hyperparameters, device visibility helper |
| `data.py` | Loads Fashion-MNIST via `tf.data` with [-1, 1] scaling |
| `model.py` | Symmetric dense encoder/decoder with configurable latent size |
| `utils.py` | PSNR metric and utility functions for compiling/saving |
| `train.py` | Scriptable training loop with checkpoint + metrics export |
| `inference.py` | Helpers for loading saved models and reconstructing batches |

---

## 3. Run it

```bash
python -m pip install tensorflow matplotlib
python "Deep Learning/Neural Networks/Autoencoders/Vanilla Autoencoder/tensorflow/src/train.py"
```

Artefacts live in `artifacts/tensorflow_vanilla_ae/` (`vanilla_autoencoder.keras`, `metrics.json`).

---

## 4. Practice prompts

1. Switch the optimiser to `AdamW` (available in `tf.keras.optimizers.experimental`) and compare convergence.
2. Increase `latent_dim` to 128 and investigate the reconstruction error curve.
3. Add Gaussian noise to the inputs inside `data.py` to prototype a denoising variant before tackling the dedicated module.
