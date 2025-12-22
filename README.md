# Audio Anomaly Detection via AnoGAN

A complete end-to-end pipeline for detecting anomalies (e.g., leaks, mechanical failures) in audio signals. This project uses Short-Time Fourier Transform (STFT) for preprocessing and a Deep Convolutional GAN (DCGAN) architecture to learn the "normal" state of audio, identifying anomalies through latent space optimization (Inverse Mapping).

## 🚀 Features
* **Audio Preprocessing**: Recursive folder processing of `.wav` files into compressed `.npz` spectrograms with custom sampling rates and segment lengths.
* **AnoGAN Architecture**: PyTorch implementation of DCGAN (Generator and Discriminator) for learning normal audio distributions.
* **Anomaly Scoring**: Multi-component scoring using **Residual Loss** (L1 distance), **Feature Loss** (intermediate discriminator features), or a combination of both.
* **Robustness Testing**: Built-in **SNR (Signal-to-Noise Ratio)** testing mode to evaluate model performance under varying Gaussian noise conditions.
* **Visualization**: Generates triplet plots (**Original | Reconstruction | Residual**) to visualize exactly where anomalies are detected in the spectrogram.
* **Advanced Logging**: Full support for CSV, JSON, TensorBoard, and **Weights & Biases (WandB)** for experiment tracking.

## 🛠 Tech Stack
* **Core**: Python 3.8+, PyTorch
* **Signal Processing**: SciPy, NumPy
* **Visualization**: Matplotlib
* **Experiment Tracking**: WandB, TensorBoard

## 📂 Project Structure
* `folder2STFT.py`: Preprocessing utility that reads WAV files, handles resampling, segments the signal, and computes STFT magnitude in Decibels (dB).
* `main_singleRun.py`: The core engine containing the DCGAN models, training loops, latent space optimization (inverse mapping), and evaluation suites.
