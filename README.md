# AnoGAN for Acoustic Leak Detection

This repository provides the basic AnoGAN model used in:

> R. Liu, T. Zayed, R. Taiwo, and J. Yang, “Learning from Silence: Unsupervised Deep Learning for Acoustic Leak Detection in Data-Scarce Water Distribution Networks,” *Reliability Engineering & System Safety*, 2026. https://doi.org/10.1016/j.ress.2026.113139

The release is intentionally limited to the model architecture and latent-space inversion required for anomaly scoring. Research data, pretrained weights, training pipelines, experimental scripts, and plotting components are not distributed.

## Method boundary

The implementation follows the method reported in the paper:

- acoustic signals are represented as single-channel log-magnitude STFT spectrograms;
- the GAN is trained exclusively on non-leak spectrograms to learn normal acoustic behavior;
- leak samples are not used for GAN training;
- an unseen spectrogram is mapped back to the latent space by optimizing its latent vector;
- latent inversion balances residual reconstruction loss and discriminator feature discrepancy with `lambda = 0.5`;
- the final anomaly score is the residual reconstruction loss selected by the paper’s ablation study.

The paper uses 64 × 64 model inputs, a latent dimension of 128, `ngf = 64`, `ndf = 64`, 100 latent-optimization steps, and a latent learning rate of `5e-3`. These values are the defaults in `anogan_model.py`.

## Included files

- `anogan_model.py`: Generator, Discriminator, latent inversion, residual anomaly scoring, and parameter counting.
- `requirements.txt`: minimum runtime dependency.

Data and pretrained weights are not included. Consequently, this repository alone does not reproduce the numerical results reported in the paper.

## Input

`optimize_latent` expects normalized tensors with shape:

```text
[batch_size, 1, 64, 64]
```

The tensors should contain log-magnitude STFT spectrograms and use the same normalization during training and inference.

The paper’s signal preprocessing uses 4096 Hz recordings divided into 10-second segments. STFT uses a Hann window, linear detrending, `nperseg = 1024`, `noverlap = 512`, and `nfft = 1024`; magnitude is converted to decibels before resizing to 64 × 64. Preprocessing code is outside this model-only release.

## Minimal model use

```python
import torch

from anogan_model import Discriminator, Generator, optimize_latent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Load weights trained exclusively on non-leak spectrograms.
# generator.load_state_dict(torch.load("G.pt", map_location=device))
# discriminator.load_state_dict(torch.load("D.pt", map_location=device))

spectrograms = torch.randn(2, 1, 64, 64, device=device).clamp(-1, 1)
result = optimize_latent(generator, discriminator, spectrograms)

print(result.anomaly_score)  # one residual anomaly score per sample
```

The commented checkpoint paths are illustrative only; no weight files are included in this repository.

## Data split required by the method

When integrating this model into a training pipeline:

1. Train the Generator and Discriminator only with non-leak spectrograms.
2. Calibrate the detection threshold using held-out normal scores. The paper uses the 75th percentile (`q = 0.75`).
3. Use mixed leak and non-leak samples only for evaluation after model and threshold selection.

## Citation

```bibtex
@article{liu2026learning,
  title   = {Learning from Silence: Unsupervised Deep Learning for Acoustic Leak Detection in Data-Scarce Water Distribution Networks},
  author  = {Liu, Rongsheng and Zayed, Tarek and Taiwo, Ridwan and Yang, Jingchao},
  journal = {Reliability Engineering \& System Safety},
  year    = {2026},
  doi     = {10.1016/j.ress.2026.113139}
}
```
