````markdown
# PyTorch Contractive Autoencoder

Encourages robustness by penalising the Jacobian of the encoder with respect to the inputs.

---

## Learning goals

- Understand the motivation behind contractive penalties and how they relate to adversarial robustness.
- Inspect how the penalty term evolves relative to reconstruction error during training.
- Experiment with activation choices and penalty weights to control local smoothness of the latent space.

---

## Implementation highlights

- The model exposes a `contractive_penalty` helper making it easy to log or reuse in notebooks.
- Training history captures the penalty magnitude so you can detect saturation or vanishing gradients.
- Modular design lets you swap the MLP for a convolutional encoder while retaining the penalty computation.

---

## 1. Notebook tour

- `notebooks/contractive_autoencoder_pytorch.ipynb` demonstrates training, tracks the contractive penalty, and visualises reconstructions.
- The notebook mirrors the same path adjustments and helper usage as the other PyTorch modules.

---

## 2. Source layout

| File | Purpose |
| ---- | ------- |
| `config.py` | Hyperparameters plus contractive penalty weight |
| `data.py` | Fashion-MNIST loaders |
| `model.py` | Encoder exposes a contractive penalty helper |
| `engine.py` | Training loop adding the penalty to reconstruction loss |
| `train.py` | Entry point for experiments |
| `inference.py` | Load checkpoints and reconstruct samples |
| `utils.py` | PSNR, seeding, and metrics persistence |

---

## 3. Run it

```bash
python -m pip install torch torchvision matplotlib
python "Deep Learning/Neural Networks/Autoencoders/Contractive Autoencoder/pytorch/src/train.py"
```

Artefacts: `artifacts/pytorch_contractive_ae/contractive_autoencoder.pt` and `metrics.json`.

---

## 4. Practice prompts

1. Increase `contractive_weight` to emphasise robustness and inspect PSNR changes.
2. Swap the activation from sigmoid to tanh and evaluate the penalty behaviour.
3. Chain the trained encoder with the denoising decoder for hybrid experiments.

````