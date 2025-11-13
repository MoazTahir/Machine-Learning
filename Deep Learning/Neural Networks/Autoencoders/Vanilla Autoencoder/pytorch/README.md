# PyTorch Vanilla Autoencoder

Train a fully-connected autoencoder on Fashion-MNIST using a minimal modular package.

---

## 1. Notebook tour

- `notebooks/vanilla_autoencoder_pytorch.ipynb` walks through loading the modular `src` package, launching training, and visualising reconstructions.
- The notebook mirrors the familiar pattern from other modules: add `../src` to `sys.path`, call `train.train()`, then rebuild a sample image with `inference.reconstruct`.

---

## 2. Source layout

| File | Purpose |
| ---- | ------- |
| `config.py` | Hyperparameters, device detection, and artifact paths |
| `data.py` | Fashion-MNIST loaders with standard normalisation |
| `model.py` | Fully-connected encoder/decoder with configurable latent size |
| `engine.py` | Training + evaluation loops returning MSE and PSNR |
| `train.py` | High-level entry point for CLI / notebook usage |
| `inference.py` | Lightweight helpers for checkpoint loading + reconstruction |
| `utils.py` | Seeding, PSNR calculation, and metric serialisation |

---

## 3. Run it

```bash
python -m pip install torch torchvision matplotlib
python "Deep Learning/Neural Networks/Autoencoders/Vanilla Autoencoder/pytorch/src/train.py"
```

Weights and metrics land in `artifacts/pytorch_vanilla_ae/` (`vanilla_autoencoder.pt`, `metrics.json`).

---

## 4. Practice prompts

1. Swap the hidden dimensions or latent size to explore reconstruction quality vs dimensionality.
2. Add dropout layers to the encoder and observe how PSNR evolves across epochs.
3. Replace the MLP with a convolutional autoencoder by editing `model.py` and compare qualitative outputs.
