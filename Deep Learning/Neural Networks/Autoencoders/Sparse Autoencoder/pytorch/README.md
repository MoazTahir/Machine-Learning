````markdown
# PyTorch Sparse Autoencoder

Impose sparsity in the latent representation using a KL divergence penalty.

---

## 1. Notebook tour

- `notebooks/sparse_autoencoder_pytorch.ipynb` follows the same workflow as other modules: configure, train, inspect sparsity/KL metrics, and reconstruct examples.
- The notebook demonstrates how to introspect average activation levels to verify that sparsity is enforced.

---

## 2. Source layout

| File | Purpose |
| ---- | ------- |
| `config.py` | Hyperparameters plus target sparsity and penalty weight |
| `data.py` | Standard Fashion-MNIST loaders |
| `model.py` | Dense encoder/decoder with sigmoid latent activations |
| `engine.py` | Training loops that add KL penalties on latent means |
| `train.py` | CLI/notebook training entry point |
| `inference.py` | Convenience helpers to load checkpoints and reconstruct |
| `utils.py` | Seeding, PSNR, KL divergence helpers, and metric saving |

---

## 3. Run it

```bash
python -m pip install torch torchvision matplotlib
python "Deep Learning/Neural Networks/Autoencoders/Sparse Autoencoder/pytorch/src/train.py"
```

Artefacts are stored in `artifacts/pytorch_sparse_ae/` (`sparse_autoencoder.pt`, `metrics.json`).

---

## 4. Practice prompts

1. Sweep `sparsity_weight` to understand the reconstruction vs sparsity trade-off.
2. Plot the average activation of the latent layer across epochs.
3. Combine this module with the denoising objective to build a sparse denoising autoencoder.

````