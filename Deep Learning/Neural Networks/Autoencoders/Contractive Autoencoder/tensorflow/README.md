````markdown
# TensorFlow Contractive Autoencoder

Sub-classed Keras model that adds an analytic contractive penalty to the encoder.

---

## 1. Notebook tour

- `notebooks/contractive_autoencoder_tensorflow.ipynb` follows the configure → train → reconstruct workflow.
- Watch the contractive penalty metric as it competes with reconstruction performance.

---

## 2. Source layout

| File | Purpose |
| ---- | ------- |
| `config.py` | Hyperparameters, contractive weight, device visibility |
| `data.py` | Fashion-MNIST loader returning `(input, target)` pairs |
| `model.py` | Custom Keras model exposing the contractive penalty |
| `train.py` | Trains with callbacks and exports metrics |
| `inference.py` | Loads weights and reconstructs batches |
| `utils.py` | PSNR metric helper and history serialization |

---

## 3. Run it

```bash
python -m pip install tensorflow matplotlib
python "Deep Learning/Neural Networks/Autoencoders/Contractive Autoencoder/tensorflow/src/train.py"
```

Artefacts: `artifacts/tensorflow_contractive_ae/contractive_autoencoder.weights.h5` and `metrics.json`.

---

## 4. Practice prompts

1. Increase `contractive_weight` to emphasise robustness and monitor PSNR.
2. Swap sigmoid activations for tanh to test the analytic penalty.
3. Visualise latent space contractions by interpolating between encoded points.

````