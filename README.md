Audio Anomaly Detection via AnoGAN
A complete end-to-end pipeline for detecting anomalies (e.g., leaks, mechanical failures) in audio signals. This project uses Short-Time Fourier Transform (STFT) for preprocessing and a Deep Convolutional GAN (DCGAN) architecture to learn the "normal" state of audio, identifying anomalies through latent space optimization (Inverse Mapping).

🚀 Features
Audio Preprocessing: Recursive folder processing of .wav files into compressed .npz spectrograms.
AnoGAN Architecture: PyTorch implementation of DCGAN for learning normal audio distributions.
Anomaly Scoring: Multi-component scoring using Residual Loss.
Robustness Testing: Built-in SNR (Signal-to-Noise Ratio) testing mode to evaluate model performance under varying noise conditions.
Visualization: Generates triplet plots (Original | Reconstruction | Residual) to visualize where anomalies are detected.
Advanced Logging: Supports CSV, JSON, TensorBoard, and Weights & Biases (WandB).

🛠 Tech Stack
Core: Python 3.8+, PyTorch
Signal Processing: SciPy, NumPy
Visualization: Matplotlib
Experiment Tracking: WandB, TensorBoard

📂 Project Structure
folder2STFT.py: Preprocessing utility to convert raw audio into spectrogram segments.
main_singleRun.py: The core engine for training, evaluation, and robustness testing.
