# Audio Anomaly Detection via AnoGAN

A complete end-to-end pipeline for detecting anomalies—such as leaks or mechanical failures—in audio signals from **water distribution networks**. This project utilizes Short-Time Fourier Transform (STFT) for preprocessing and a Deep Convolutional GAN (DCGAN) architecture to learn the "normal" state of acoustic environments, identifying deviations through latent space optimization (Inverse Mapping).

## 🚀 Features
* **Audio Preprocessing**: Recursive processing of `.wav` files into compressed `.npz` spectrograms with configurable sampling rates, segmentation, and normalization.
* **AnoGAN Architecture**: A robust PyTorch implementation of DCGAN (Generator and Discriminator) optimized for learning complex audio distributions.
* **Multi-Component Scoring**: Flexibility to calculate anomaly scores using **Residual Loss** (L1 distance in image space), **Feature Loss** (discrepancies in discriminator feature maps), or a weighted combination of both.
* **Robustness Testing**: Integrated SNR testing suite to evaluate model reliability under varying levels of environmental Gaussian noise.
* **Insightful Visualization**: Automatically generates triplet plots (**Original | Reconstruction | Residual**) to pinpoint exactly where anomalies occur within the frequency spectrum.
* **Comprehensive Logging**: Full integration with CSV, JSON, TensorBoard, and **Weights & Biases (WandB)** for professional-grade experiment tracking.

## 🛠 Tech Stack
* **Core**: Python 3.8+, PyTorch
* **Signal Processing**: SciPy, NumPy
* **Visualization**: Matplotlib
* **Experiment Tracking**: WandB, TensorBoard

## 📂 Project Structure
* `folder2STFT.py`: A high-performance preprocessing utility that handles resampling, signal segmentation, and STFT magnitude computation in Decibels (dB).
* `main_singleRun.py`: The central execution engine containing the GAN architecture, training loops, latent space "inverse mapping" logic, and the evaluation suite.
