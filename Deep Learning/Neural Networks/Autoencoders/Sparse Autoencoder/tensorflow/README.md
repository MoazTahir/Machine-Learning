````markdown
# TensorFlow Sparse Autoencoder

Custom training loop built with Keras to impose a KL divergence sparsity constraint on the latent code.

---

## Learning goals

- Observe how sparsity constraints manifest in a custom `train_step` for subclassed Keras models.
- Monitor KL penalty, reconstruction loss, and PSNR together to understand competing objectives.
- Practise tuning sparsity targets and penalty weights using the config-driven workflow.

---

## Implementation highlights

- The model overrides `train_step` and `test_step`, so the KL penalty is baked into every optimisation update.
- Metrics are registered as class attributes, ensuring they appear automatically in `model.fit` logs.
- Checkpoints store only weights, making it easy to reload into compatible architectures for transfer experiments.

---

## 1. Notebook tour

- `notebooks/sparse_autoencoder_tensorflow.ipynb` mirrors the configure → train → reconstruct workflow.
- The notebook highlights how the KL penalty and PSNR metrics evolve through training.

---

## 2. Source layout

| File | Purpose |
| ---- | ------- |
| `config.py` | Hyperparameters, sparsity settings, and device visibility |
| `data.py` | Builds `(input, target)` pairs from Fashion-MNIST via `tf.data` |
| `model.py` | Subclassed `tf.keras.Model` with custom train/test steps |
| `train.py` | High-level entry point that compiles, trains, and logs metrics |
| `inference.py` | Utilities for loading weights and reconstructing batches |
| `utils.py` | KL divergence helper, PSNR metric, and metrics serialization |

---

## 3. Run it

```bash
python -m pip install tensorflow matplotlib
python "Deep Learning/Neural Networks/Autoencoders/Sparse Autoencoder/tensorflow/src/train.py"
```

Checkpoint weights land in `artifacts/tensorflow_sparse_ae/` (`sparse_autoencoder.weights.h5`, `metrics.json`).

---

## 4. Practice prompts

1. Reduce `sparsity_target` to 0.02 and observe the impact on PSNR.
2. Swap the latent activation to `relu` and inspect how the KL penalty behaves.
3. Combine the sparse penalty with denoising by perturbing inputs inside `data.py`.

````