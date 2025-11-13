````markdown
# TensorFlow Denoising Autoencoder

Mirror of the PyTorch denoising module using Keras and `tf.data`.

---

## Learning goals

- Learn how to generate noisy/clean pairs inside a `tf.data` pipeline.
- Track PSNR alongside reconstruction loss to judge qualitative improvements.
- Practice restoring checkpoints and running inference to denoise arbitrary batches.

---

## Implementation highlights

- Device visibility is configured up front so you can target MPS, GPU, or CPU with a single setting.
- The custom PSNR metric is registered during compilation, making it available in both training logs and callbacks.
- Checkpoints save the full model (`.keras`) artefact so you can reload without rebuilding the graph.

---

## 1. Notebook tour

- `notebooks/denoising_autoencoder_tensorflow.ipynb` mirrors the configure → train → reconstruct flow with the TensorFlow stack.
- The notebook showcases how the input pipeline injects Gaussian noise and how PSNR evolves during training.

---

## 2. Source layout

| File | Purpose |
| ---- | ------- |
| `config.py` | Device visibility, hyperparameters, and noise level configuration |
| `data.py` | Builds noisy/clean pairs from Fashion-MNIST using `tf.data` |
| `model.py` | Symmetric dense encoder/decoder identical to the PyTorch variant |
| `utils.py` | Custom PSNR metric plus compile/save helpers |
| `train.py` | High-level training entry point with checkpointing |
| `inference.py` | Convenience helpers to load a saved model and denoise batches |

---

## 3. Run it

```bash
python -m pip install tensorflow matplotlib
python "Deep Learning/Neural Networks/Autoencoders/Denoising Autoencoder/tensorflow/src/train.py"
```

Artefacts land in `artifacts/tensorflow_denoising_ae/` (`denoising_autoencoder.keras`, `metrics.json`).

---

## 4. Practice prompts

1. Increase `noise_std` in `config.py` and measure the PSNR drop-off.
2. Replace the dense stack with convolutions by editing `model.py`.
3. Export denoised samples back to disk for qualitative inspection.

````