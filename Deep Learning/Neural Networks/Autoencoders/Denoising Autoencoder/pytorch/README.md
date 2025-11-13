````markdown
# PyTorch Denoising Autoencoder

Learn to undo Gaussian noise with a fully-connected denoising autoencoder built on top of the vanilla package.

---

## Learning goals

- Understand how input corruption changes the training objective compared with the vanilla autoencoder.
- Observe how PSNR and reconstruction loss respond as you sweep the injected noise level.
- Reuse modular training utilities to extend the baseline autoencoder with minimal code changes.

---

## Implementation highlights

- Noise injection happens inside a dataset wrapper so the model only ever sees corrupted inputs.
- Training history tracks both MSE and PSNR to quantify perceptual fidelity after denoising.
- Checkpoints and metrics are written to a dedicated artifact directory for quick experiment comparisons.

---

## 1. Notebook tour

- `notebooks/denoising_autoencoder_pytorch.ipynb` demonstrates how to add Gaussian noise, launch training, and visualise before/after reconstructions.
- The flow mirrors the other modules: append `../src` to `sys.path`, call `train.train()`, then sample reconstructions via `inference.denoise`.

---

## 2. Source layout

| File | Purpose |
| ---- | ------- |
| `config.py` | Hyperparameters, noise level, and artifact paths |
| `data.py` | Fashion-MNIST loaders that inject Gaussian corruption on the fly |
| `model.py` | Symmetric MLP encoder/decoder reused from the vanilla module |
| `engine.py` | Training loops that track reconstruction MSE + PSNR |
| `train.py` | High-level entry point for running experiments |
| `inference.py` | Helper for loading checkpoints and denoising arbitrary batches |
| `utils.py` | Seed control, PSNR utility, and metric serialisation |

---

## 3. Run it

```bash
python -m pip install torch torchvision matplotlib
python "Deep Learning/Neural Networks/Autoencoders/Denoising Autoencoder/pytorch/src/train.py"
```

Weights and metrics land in `artifacts/pytorch_denoising_ae/` (`denoising_autoencoder.pt`, `metrics.json`).

---

## 4. Practice prompts

1. Sweep `noise_std` in `config.py` and monitor how PSNR scales with heavier corruption.
2. Swap the MLP for a convolutional architecture and compare denoising quality.
3. Try curriculum learning: start with low noise and gradually ramp it up during training.

````