import argparse
import csv
import json
import random
import sys
import time
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# Set backend to Agg to avoid GUI errors on servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================
# 0) Utility Functions & Noise Helpers
# =========================

def abspath_or_none(p):
    """Convert path to absolute string; keep None/empty string as None."""
    if p is None or (isinstance(p, str) and p.strip() == ""):
        return None
    return str(Path(p).expanduser().resolve())


def ensure_dir_abs(p: str) -> Path:
    """Convert p to absolute path and create directory (parents=True, exist_ok=True)."""
    try:
        p_abs = Path(p).expanduser().resolve()
        p_abs.mkdir(parents=True, exist_ok=True)
        return p_abs
    except Exception as e:
        print(f"[WARN] Cannot create directory {p} -> {e}. Falling back to temp directory.")
        fallback = Path(tempfile.gettempdir()) / "anogan_fallback"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def set_seed(seed=42):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def try_import_wandb(enabled: bool):
    """Try importing wandb; return None if not enabled or import fails."""
    if not enabled:
        return None
    try:
        import wandb
        return wandb
    except ImportError:
        print("[INFO] wandb not installed. Continuing without wandb logging.")
    except Exception as e:
        print(f"[WARN] wandb unavailable: {e}")
    return None


def add_gaussian_noise(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Add Gaussian noise with specified SNR (dB) to input tensor x.
    Formula: SNR_db = 10 * log10(P_signal / P_noise)
    """
    # Calculate signal power (Mean Square)
    sig_power = torch.mean(x ** 2)

    if sig_power == 0:
        return x

    # Calculate required noise power
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise_std = torch.sqrt(noise_power)

    # Generate and add noise
    noise = torch.randn_like(x) * noise_std
    return x + noise


# =========================
# 1) Dataset (Reading 'S_db')
# =========================
class SpectrogramNPZDataset(Dataset):
    """
    Dataset class: Reads .npz files under a directory.
    Expected structure:
        data_dir/
            ├── ClassA/
            │   ├── 1.npz
            │   └── ...
            └── ClassB/
                ├── ...

    Each npz file must contain the key 'S_db'.
    """

    def __init__(self, data_dir, classes=("Noleak",),
                 clip_db=(-120.0, 0.0), zscore=True, add_channel_dim=True,
                 resize_to: Tuple[int, int] = (64, 64)):
        self.root = Path(data_dir)
        self.classes = list(classes)
        self.clip_db = clip_db
        self.zscore = zscore
        self.add_channel_dim = add_channel_dim
        self.resize_to = resize_to

        # Label map: Assuming binary classification
        self.label_map = {
            "Leak": torch.tensor([0., 1.], dtype=torch.float32),  # Anomaly/Leak (Positive)
            "Noleak": torch.tensor([1., 0.], dtype=torch.float32),  # Normal/Noleak (Negative)
        }
        self.samples = []

        def _rglob_npz(d: Path):
            return list(d.rglob("*.npz"))

        # Check root directory
        if not self.root.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.root}")

        # Scan files
        if len(self.classes) == 1:
            # Single class mode (usually for training set, containing only normal samples)
            only_cls = self.classes[0]
            cdir = self.root / only_cls
            files = []
            if cdir.exists():
                files = _rglob_npz(cdir)

            # If not found in class directory, try recursive search in root
            if not files:
                # print(f"[INFO] Files not found in {cdir}, attempting global search in {self.root}...")
                files = _rglob_npz(self.root)

            if not files:
                raise RuntimeError(f"No .npz files found for class '{only_cls}' under {self.root}")

            for p in files:
                self.samples.append({"path": p, "cls": only_cls})
        else:
            # Multi-class mode (usually for test set)
            for cls in self.classes:
                cdir = self.root / cls
                if not cdir.exists():
                    print(f"[WARN] Missing class directory: {cdir} (Skipping this class)")
                    continue
                files = _rglob_npz(cdir)
                for p in files:
                    self.samples.append({"path": p, "cls": cls})

        if not self.samples:
            raise RuntimeError(f"No .npz files found under {self.root} for classes: {self.classes}")

        # Read one sample to determine shape
        self._example_shape = None
        for s in self.samples[:5]:  # Try the first few in case of corruption
            try:
                with np.load(s["path"]) as npz:
                    if "S_db" in npz:
                        a = np.asarray(npz["S_db"], np.float32)
                        self._example_shape = (1, *a.shape) if a.ndim == 2 else tuple(a.shape)
                        break
            except Exception:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            with np.load(s["path"]) as npz:
                if "S_db" not in npz.files:
                    raise KeyError(f"{s['path']} does not contain 'S_db' key")
                spec = np.asarray(npz["S_db"], dtype=np.float32)
        except Exception as e:
            # If reading fails, return dummy data to prevent crash (or raise)
            print(f"[ERR] Failed to read {s['path']}: {e}")
            spec = np.zeros(self.resize_to, dtype=np.float32)

        # 1. Clip
        if self.clip_db is not None:
            lo, hi = self.clip_db
            spec = np.clip(spec, lo, hi)

        # 2. Normalize (Z-Score)
        if self.zscore:
            m, sd = spec.mean(), spec.std()
            spec = (spec - m) / (sd + 1e-6)

        # 3. Add channel dimension [H, W] -> [1, H, W]
        if self.add_channel_dim and spec.ndim == 2:
            spec = spec[None, ...]

        x = torch.from_numpy(spec)

        # 4. Resize -> [1, 64, 64]
        if self.resize_to is not None:
            Ht, Wt = self.resize_to
            x = F.interpolate(x.unsqueeze(0), size=(Ht, Wt),
                              mode='bilinear', align_corners=False).squeeze(0)

        # 5. Get Label
        if len(self.classes) == 1:
            y = torch.tensor(0, dtype=torch.long)  # Usually all 0 for training
        else:
            y = self.label_map.get(s["cls"], torch.tensor(0, dtype=torch.long))

        return x, y

    def class_counts(self):
        return dict(Counter(s["cls"] for s in self.samples))


# =========================
# 2) DCGAN Model Structure (G/D)
# =========================
class Generator(nn.Module):
    def __init__(self, z_dim=128, ngf=64, out_ch=1, target_size=(64, 64)):
        super().__init__()
        self.target_size = target_size
        # Input: [z_dim, 1, 1]
        self.net = nn.Sequential(
            # block 1: 4x4
            nn.ConvTranspose2d(z_dim, ngf * 8, (4, 4), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            # block 2: 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            # block 3: 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # block 4: 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            # block 5: 64x64
            nn.ConvTranspose2d(ngf, out_ch, (4, 4), (2, 2), (1, 1), bias=False),
            nn.Tanh()
        )
        # Force resize to target size (compatibility for different input sizes)
        self.resize = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)

    def forward(self, z):
        out = self.net(z)
        out = self.resize(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, ndf=64, in_ch=1, return_features=True):
        super().__init__()
        self.return_features = return_features
        # 64x64 -> 32x32
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 32x32 -> 16x16
        self.b2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 16x16 -> 8x8 (Feature Extraction Layer)
        self.b3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 8x8 -> 4x4
        self.b4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False)

    def forward(self, x):
        h1 = self.b1(x)
        h2 = self.b2(h1)
        h3 = self.b3(h2)  # Intermediate features
        h4 = self.b4(h3)
        pooled = self.global_pool(h4)
        logits = self.out(pooled)
        prob = torch.sigmoid(logits)

        # AnoGAN requires intermediate features to calculate Feature Loss
        if self.return_features:
            return prob, h3
        else:
            return prob


# =========================
# 3) Logging & Visualization
# =========================
class TrainRecorder:
    def __init__(self, log_dir: Path, use_tb: bool = True, wandb_run=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.hist = []
        self.tb = None
        self.wandb_run = wandb_run
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("[INFO] TensorBoard not installed, skipping TB logging.")

    def log_epoch(self, epoch, d_loss, g_loss, iters, elapsed_s):
        row = {"epoch": int(epoch), "d_loss": float(d_loss), "g_loss": float(g_loss),
               "iters": int(iters), "elapsed_s": float(elapsed_s)}
        self.hist.append(row)
        if self.tb:
            self.tb.add_scalar("loss/D", d_loss, epoch)
            self.tb.add_scalar("loss/G", g_loss, epoch)
            self.tb.add_scalar("time/epoch_s", elapsed_s, epoch)
        if self.wandb_run:
            self.wandb_run.log({"loss/D": d_loss, "loss/G": g_loss, "time/epoch_s": elapsed_s, "epoch": epoch})

    def save_csv(self):
        csv_path = self.log_dir / "train_log.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "d_loss", "g_loss", "iters", "elapsed_s"])
            w.writeheader()
            w.writerows(self.hist)
        print(f"[LOG] CSV saved -> {csv_path}")

    def save_json(self):
        json_path = self.log_dir / "train_log.json"
        with open(json_path, "w") as f:
            json.dump(self.hist, f, indent=2)

    def plot_png(self):
        if not self.hist: return
        ep = [r["epoch"] for r in self.hist]
        d = [r["d_loss"] for r in self.hist]
        g = [r["g_loss"] for r in self.hist]
        plt.figure(figsize=(7, 4))
        plt.plot(ep, d, label="D Loss")
        plt.plot(ep, g, label="G Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("AnoGAN Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.log_dir / "train_loss.png", dpi=150)
        plt.close()

    def close(self):
        if self.tb: self.tb.close()


def _to_np_img(x):
    """Convert Tensor to numpy image format (H, W) for plotting"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float()
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
    return x.numpy()


def save_triplet_vis(x, x_hat, title, out_path, wandb_run=None, wandb_key=None, wandb_caption=""):
    """
    Generate triplet plot: Input | Reconstruction | Residual
    """
    X = _to_np_img(x)
    Xh = _to_np_img(x_hat)
    R = np.abs(X - Xh)

    # Coordinates: Time 0-10s, Freq 0-2048Hz (Adjust based on physical meaning)
    extent = [0, 10, 0, 2048]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)

    im0 = axes[0].imshow(X, aspect='auto', origin='lower', extent=extent)
    axes[0].set_title("Input S_db")

    im1 = axes[1].imshow(Xh, aspect='auto', origin='lower', extent=extent)
    axes[1].set_title("G(z*) Recon")

    im2 = axes[2].imshow(R, aspect='auto', origin='lower', extent=extent)
    axes[2].set_title("|Residual|")

    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    fig.suptitle(title, y=1.02)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)

    # WandB logging
    if wandb_run and wandb_key:
        try:
            import wandb
            wandb_run.log({wandb_key: wandb.Image(fig, caption=wandb_caption)})
        except Exception:
            pass
    plt.close(fig)


@torch.no_grad()
def _pick_items(ds, k=4, seed=0):
    n = len(ds)
    rng = np.random.default_rng(seed)
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    return idx.tolist()


# =========================
# 4) Latent Space Optimization (AnoGAN Scoring)
# =========================
def extract_feat(D: nn.Module, x: torch.Tensor):
    prob, feat = D(x)
    return feat


def optimise_z_for_x(G: nn.Module, D: nn.Module, x: torch.Tensor,
                     z_dim=128, steps=200, lr=1e-2, lambda_feat=0.9):
    """
    Inverse Mapping: Find best latent vector z* such that G(z*) is closest to input x.
    Loss = (1 - lambda) * Residual_Loss + lambda * Feature_Loss
    """
    G.eval()
    D.eval()
    # Extract features of real sample
    feat_x = extract_feat(D, x).detach()

    # Initialize z
    z = torch.randn(x.size(0), z_dim, 1, 1, device=x.device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)

    for _ in range(steps):
        x_hat = G(z)
        _, feat_hat = D(x_hat)

        loss_res = F.l1_loss(x_hat, x, reduction='mean')
        loss_fm = F.l1_loss(feat_hat, feat_x, reduction='mean')

        # Combined Loss
        loss = (1 - lambda_feat) * loss_res + lambda_feat * loss_fm

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Final Loss calculation (No grad)
    with torch.no_grad():
        x_hat = G(z)
        _, feat_hat = D(x_hat)
        loss_res = F.l1_loss(x_hat, x, reduction='mean')
        loss_fm = F.l1_loss(feat_hat, feat_x, reduction='mean')
        loss = (1 - lambda_feat) * loss_res + lambda_feat * loss_fm

    return z.detach(), x_hat.detach(), float(loss_res.item()), float(loss_fm.item()), float(loss.item())


def visualise_anogan(ds, G, D, device, k=4, indices=None, out_dir=None, tag="epoch0000", seed=0,
                     z_steps=200, z_lr=1e-2, lambda_feat=0.9, wandb_run=None):
    if out_dir is None:
        out_dir = Path("runs/viz")
    out_dir = ensure_dir_abs(str(out_dir))

    # Select indices to visualize
    if indices is not None:
        idx_list = indices
    else:
        idx_list = _pick_items(ds, k=k, seed=seed)

    if not idx_list:
        return

    run_dir = ensure_dir_abs(str(out_dir / str(tag)))

    for i, idx in enumerate(idx_list):
        x, y_val = ds[idx]  # x: [1,H,W]

        # Get label string
        lbl_str = "unk"
        if isinstance(y_val, torch.Tensor) and y_val.numel() == 2:
            lbl_str = "Leak" if y_val[1] > 0.5 else "Noleak"

        xb = x.unsqueeze(0).to(device)

        # Perform Inverse Mapping
        with torch.enable_grad():
            _, xh, _, _, _ = optimise_z_for_x(
                G, D, xb, steps=z_steps, lr=z_lr, lambda_feat=lambda_feat
            )
        xh_cpu = xh.cpu().squeeze(0)

        title = f"idx{idx}_{lbl_str}"
        out_path = run_dir / f"{title}.png"

        save_triplet_vis(x, xh_cpu, title, out_path,
                         wandb_run=wandb_run,
                         wandb_key=f"{tag}/sample_{i}",
                         wandb_caption=f"idx={idx} ({lbl_str})")


# =========================
# 5) Splitting & Scoring
# =========================
def split_normal_dataset(train_dir, calib_ratio=0.2, seed=42, resize_to=(64, 64)):
    """Split normal dataset into: Training set (Train) and Calibration set (Calib)."""
    full_ds = SpectrogramNPZDataset(train_dir, classes=("Noleak",), resize_to=resize_to)
    n = len(full_ds)
    idx_all = list(range(n))
    random.Random(seed).shuffle(idx_all)

    n_calib = int(round(n * calib_ratio))
    calib_idx = idx_all[:n_calib]
    train_idx = idx_all[n_calib:]

    return Subset(full_ds, train_idx), Subset(full_ds, calib_idx), full_ds


def anogan_scores_on_dataset(ds, G, D, device="cuda", max_items=None,
                             z_steps=200, z_lr=1e-2, lambda_feat=0.9,
                             score_combo="sum", snr_db: Optional[float] = None):
    """
    Calculate anomaly scores for each sample in the dataset.
    Supports 'snr_db' parameter to inject noise during inference for robustness testing.
    """
    G.eval()
    D.eval()
    scores, labels, res_list, fm_list = [], [], [], []
    n = len(ds) if max_items is None else min(max_items, len(ds))

    noise_msg = f" (SNR={snr_db}dB)" if snr_db is not None else ""
    print(f"[SCORING] Processing {n} samples{noise_msg}...")

    for i in range(n):
        x, y = ds[i]
        xb = x.unsqueeze(0).to(device)  # [1, 1, H, W]

        # [NEW] Inject noise
        if snr_db is not None:
            xb = add_gaussian_noise(xb, snr_db)

        # Inverse Mapping
        _, _, loss_res, loss_fm, loss_total = optimise_z_for_x(
            G, D, xb, steps=z_steps, lr=z_lr, lambda_feat=lambda_feat
        )

        # Select scoring method
        if score_combo == "residual":
            s = loss_res
        elif score_combo == "feature":
            s = loss_fm
        else:
            s = loss_total  # sum

        scores.append(float(s))
        res_list.append(float(loss_res))
        fm_list.append(float(loss_fm))

        # Parse Label
        lbl = 0
        if isinstance(y, torch.Tensor) and y.ndim == 1 and y.numel() == 2:
            lbl = int(y[1].item() > 0.5)
        labels.append(lbl)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n}")

    return np.array(scores, np.float32), np.array(labels, np.int32), \
        np.array(res_list, np.float32), np.array(fm_list, np.float32)


def compute_metrics_from_scores(scores: np.ndarray, labels: np.ndarray, tau: float) -> Dict[str, Any]:
    """Compute metrics based on given threshold tau."""
    preds = (scores > tau).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = tp + tn + fp + fn

    acc = (tp + tn) / total if total > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Alarm Rate

    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)

    auroc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(labels)) > 1:
            auroc = roc_auc_score(labels, scores)
    except ImportError:
        pass

    return {
        "AUROC": float(auroc), "F1": float(f1), "Acc": float(acc),
        "Precision": float(p), "Recall": float(r), "FAR": float(far),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn, "tau": float(tau),
    }


# =========================
# 6) Training Loop
# =========================
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters(): p.requires_grad_(flag)


def train_anogan(train_source, batch=64, epochs=200, device="cuda",
                 n_workers=4, pin_memory=True, z_dim=128, ngf=64, ndf=64,
                 lr_d=2e-4, lr_g=2e-4, betas=(0.5, 0.999),
                 log_dir="./runs", save_every=10, use_tb=True, viz_every=10,
                 wandb_run=None, args=None):
    # Prepare Dataset
    if isinstance(train_source, (str, Path)):
        ds = SpectrogramNPZDataset(train_source, classes=("Noleak",), resize_to=args.ImageSize)
        vis_ds = ds
    else:
        ds = train_source
        vis_ds = ds

    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=n_workers,
                    pin_memory=pin_memory, drop_last=True)

    # Initialize Models
    G = Generator(z_dim=z_dim, ngf=ngf, out_ch=1, target_size=args.ImageSize).to(device)
    D = Discriminator(ndf=ndf, in_ch=1, return_features=True).to(device)

    optD = torch.optim.Adam(D.parameters(), lr=lr_d, betas=betas)
    optG = torch.optim.Adam(G.parameters(), lr=lr_g, betas=betas)
    bce = nn.BCELoss()

    log_dir_abs = ensure_dir_abs(abspath_or_none(log_dir))
    rec = TrainRecorder(log_dir=log_dir_abs, use_tb=use_tb, wandb_run=wandb_run)

    print(f"[TRAIN] Start training: {epochs} Epochs, Device: {device}, LogDir: {log_dir_abs}")
    try:
        for ep in range(1, epochs + 1):
            t0 = time.time()
            iters = 0
            d_running, g_running = 0.0, 0.0
            G.train()
            D.train()

            for xb, _ in dl:
                iters += 1
                xb = xb.to(device)
                bsize = xb.size(0)

                real_label = torch.ones(bsize, 1, 1, 1, device=device)
                fake_label = torch.zeros(bsize, 1, 1, 1, device=device)

                # --- Train Discriminator ---
                set_requires_grad(D, True)
                set_requires_grad(G, False)

                # Real
                prob_r, _ = D(xb)
                loss_D_real = bce(prob_r, real_label)

                # Fake
                z = torch.randn(bsize, z_dim, 1, 1, device=device)
                x_fake = G(z).detach()
                prob_f, _ = D(x_fake)
                loss_D_fake = bce(prob_f, fake_label)

                loss_D = (loss_D_real + loss_D_fake) * 0.5
                optD.zero_grad(set_to_none=True)
                loss_D.backward()
                optD.step()

                # --- Train Generator ---
                set_requires_grad(D, False)
                set_requires_grad(G, True)

                z = torch.randn(bsize, z_dim, 1, 1, device=device)
                x_fake = G(z)
                prob_f, _ = D(x_fake)
                # G's goal is to fool D into thinking it's Real
                loss_G = bce(prob_f, real_label)

                optG.zero_grad(set_to_none=True)
                loss_G.backward()
                optG.step()

                d_running += loss_D.item()
                g_running += loss_G.item()

            elapsed = time.time() - t0
            rec.log_epoch(ep, d_running / iters, g_running / iters, iters, elapsed)

            if ep % 5 == 0 or ep == 1:
                print(f"[Ep {ep:03d}] D_loss={d_running / iters:.4f} G_loss={g_running / iters:.4f} ({elapsed:.1f}s)")

            # Visualization
            if viz_every and (ep % viz_every == 0 or ep == 1 or ep == epochs):
                visualise_anogan(vis_ds, G, D, device, k=args.viz_k, out_dir=log_dir_abs / "viz",
                                 tag=f"epoch_{ep:04d}_inv", seed=ep,
                                 z_steps=min(150, args.z_steps), z_lr=args.z_lr,
                                 lambda_feat=args.lambda_feat, wandb_run=wandb_run)

            # Save Checkpoint
            if (save_every is not None) and (ep % save_every == 0 or ep == epochs):
                ckpt_dir = ensure_dir_abs(str(log_dir_abs / "checkpoints"))
                path = ckpt_dir / f"ckpt_e{ep:04d}.pt"
                torch.save({"epoch": ep, "G": G.state_dict(), "D": D.state_dict()}, path)
                print(f"[CKPT] Saved {path}")

    finally:
        rec.save_csv()
        rec.save_json()
        rec.plot_png()
        rec.close()
        # Save final weights
        final_dir = ensure_dir_abs(str(log_dir_abs / "final_weights"))
        torch.save(G.state_dict(), final_dir / "G.pt")
        torch.save(D.state_dict(), final_dir / "D.pt")

    return G, D


# =========================
# 7) Main Program & CLI
# =========================
def single_run_train(args, wandb_run=None):
    """Logic for Training Mode"""
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    set_seed(args.seed)

    # 1. If calib_dir is not specified, automatically split from train_dir
    if args.calib_dir is None:
        train_sub, calib_sub, _ = split_normal_dataset(
            args.train_dir, args.calib_ratio, args.seed, args.ImageSize
        )
        print(f"[SPLIT] Train set: {len(train_sub)}, Calibration set: {len(calib_sub)}")
        # Train
        G, D = train_anogan(
            train_sub, batch=args.batch, epochs=args.epochs, device=device,
            log_dir=args.log_dir, z_dim=args.z_dim, ngf=args.ngf, ndf=args.ndf,
            wandb_run=wandb_run, args=args
        )
        calib_ds = calib_sub
    else:
        # Train
        G, D = train_anogan(
            args.train_dir, batch=args.batch, epochs=args.epochs, device=device,
            log_dir=args.log_dir, z_dim=args.z_dim, ngf=args.ngf, ndf=args.ndf,
            wandb_run=wandb_run, args=args
        )
        calib_ds = SpectrogramNPZDataset(args.calib_dir, classes=("Noleak",), resize_to=args.ImageSize)

    # 2. Calibrate (Compute Threshold Tau)
    print("\n[CALIB] Computing threshold on calibration set...")
    calib_scores, _, _, _ = anogan_scores_on_dataset(
        calib_ds, G, D, device=device, max_items=args.max_items,
        z_steps=args.z_steps, z_lr=args.z_lr, lambda_feat=args.lambda_feat,
        score_combo=args.score_combo
    )
    # Determine threshold based on quantile
    tau = float(np.quantile(calib_scores, args.quantile))
    print(f"[CALIB] Quantile {args.quantile} -> Tau {tau:.6f}")

    # Save Tau
    calib_log = Path(args.log_dir) / "calibration"
    calib_log.mkdir(parents=True, exist_ok=True)
    with open(calib_log / "tau.txt", "w") as f:
        f.write(f"tau={tau}\n")
    if wandb_run: wandb_run.log({"calib/tau": tau})

    return tau


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Mode selection
    ap.add_argument("--mode", choices=["train", "eval", "test_snr"], default="eval", help="Run mode")

    # Path Configuration (De-identified)
    g_path = ap.add_argument_group("Path Configuration")
    g_path.add_argument("--train_dir", default="./data/train", help="Normal sample training set path (Noleak)")
    g_path.add_argument("--test_dir", default="./data/test", help="Test set path (Containing Leak/Noleak)")
    g_path.add_argument("--calib_dir", default=None,
                        help="Calibration set path (Optional, defaults to split from training set)")
    g_path.add_argument("--log_dir", default="./runs/anogan_exp", help="Directory for logs and outputs")

    # Model Params
    g_model = ap.add_argument_group("Model Params")
    g_model.add_argument("--device", default="cuda", help="Computation device (cpu/cuda)")
    g_model.add_argument("--z_dim", type=int, default=128, help="Latent vector dimension")
    g_model.add_argument("--ngf", type=int, default=32, help="Generator conv channel base")
    g_model.add_argument("--ndf", type=int, default=32, help="Discriminator conv channel base")
    g_model.add_argument("--ImageSize", type=int, nargs=2, default=(64, 64), help="Image size (H W)")

    # Training Params
    g_train = ap.add_argument_group("Training Params")
    g_train.add_argument("--epochs", type=int, default=100)
    g_train.add_argument("--batch", type=int, default=64)
    g_train.add_argument("--lr_d", type=float, default=4e-4)
    g_train.add_argument("--lr_g", type=float, default=1e-4)
    g_train.add_argument("--seed", type=int, default=42)
    g_train.add_argument("--viz_every", type=int, default=10, help="Visualize every N epochs")

    # Eval/Inference Params
    g_eval = ap.add_argument_group("Eval/Scoring Params")
    g_eval.add_argument("--calib_ratio", type=float, default=0.2, help="Ratio of training set to use for calibration")
    g_eval.add_argument("--quantile", type=float, default=0.9,
                        help="Quantile for threshold (e.g., 0.9 excludes 10% outliers)")
    g_eval.add_argument("--z_steps", type=int, default=200, help="Inverse mapping optimization steps")
    g_eval.add_argument("--z_lr", type=float, default=5e-3, help="Inverse mapping learning rate")
    g_eval.add_argument("--lambda_feat", type=float, default=0.5, help="Weight for Feature Loss")
    g_eval.add_argument("--score_combo", default="residual", choices=["sum", "residual", "feature"],
                        help="Method to calculate anomaly score")
    g_eval.add_argument("--max_items", type=int, default=None, help="Test only first N samples (for debugging)")

    # Checkpoint Paths (For Eval)
    g_ckpt = ap.add_argument_group("Checkpoint Paths (For Eval)")
    g_ckpt.add_argument("--ckpt", default=None, help="Load specific Checkpoint (.pt)")
    g_ckpt.add_argument("--ckpt_g", default="./runs/anogan_exp/final_weights/G.pt", help="Generator weights path")
    g_ckpt.add_argument("--ckpt_d", default="./runs/anogan_exp/final_weights/D.pt", help="Discriminator weights path")
    g_ckpt.add_argument("--tau_path", default=None, help="Specify text file containing tau value")

    # SNR Robustness Test Params
    g_snr = ap.add_argument_group("SNR Robustness Test")
    g_snr.add_argument("--snr_levels", type=float, nargs="+", default=[30, 20, 10, 5, 0, -5],
                       help="List of SNR levels to test (dB)")
    g_snr.add_argument("--test_quantiles", type=float, nargs="+", default=[0.8, 0.9, 0.95, 0.99],
                       help="Different quantile thresholds to test")

    # WandB
    g_wandb = ap.add_argument_group("Logging")
    g_wandb.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    g_wandb.add_argument("--wandb_project", default="anogan-public", help="WandB Project Name")
    g_wandb.add_argument("--wandb_run_name", default="run", help="WandB Run Name")

    args = ap.parse_args()

    # ==========================================
    # === Shared Logic: Load Model (Eval & Test SNR) ===
    # ==========================================
    if args.mode in ["eval", "test_snr"]:
        device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
        set_seed(args.seed)
        print(f"\n[INFO] Mode: {args.mode.upper()} on {device}")

        # 1. Load Model Structure
        G = Generator(z_dim=args.z_dim, ngf=args.ngf, out_ch=1, target_size=args.ImageSize).to(device)
        D = Discriminator(ndf=args.ndf, in_ch=1, return_features=True).to(device)

        loaded = False
        # Prioritize loading full checkpoint
        if args.ckpt and Path(args.ckpt).exists():
            print(f"[LOAD] Loading checkpoint: {args.ckpt}")
            ckpt = torch.load(args.ckpt, map_location=device)
            G.load_state_dict(ckpt["G"])
            D.load_state_dict(ckpt["D"])
            loaded = True
        # Otherwise load separate weights
        elif args.ckpt_g and Path(args.ckpt_g).exists():
            print(f"[LOAD] Loading weights G:{args.ckpt_g}, D:{args.ckpt_d}")
            G.load_state_dict(torch.load(args.ckpt_g, map_location=device))
            D.load_state_dict(torch.load(args.ckpt_d, map_location=device))
            loaded = True

        if not loaded:
            print("[ERR] Failed to load weights. Please check path config (--ckpt or --ckpt_g/d).")
            sys.exit(1)

        # 2. Prepare Dataset (Validation/Test set usually contains Leak and Noleak)
        # Ensure path exists
        if not Path(args.test_dir).exists():
            print(f"[ERR] Test set directory does not exist: {args.test_dir}")
            print("Please specify correct data path using --test_dir parameter.")
            sys.exit(1)

        test_ds = SpectrogramNPZDataset(args.test_dir, classes=("Leak", "Noleak"), resize_to=args.ImageSize)

        # WandB Init
        wrun = None
        if args.wandb:
            wrun = try_import_wandb(True)
            if wrun:
                wrun.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # ==========================================
    # === Mode: TEST SNR (Robustness Test) ===
    # ==========================================
    if args.mode == "test_snr":
        print(f"\n[INFO] Starting SNR Robustness Test...")
        print(f"   SNR Levels (dB): {args.snr_levels}")
        print(f"   Quantiles      : {args.test_quantiles}")

        # 1. Calculate Calibration Score on Clean Data (Clean Baseline)
        print("\n[INFO] Computing clean calibration set scores (Baseline)...")
        calib_ds = None
        if args.calib_dir:
            calib_ds = SpectrogramNPZDataset(args.calib_dir, classes=("Noleak",), resize_to=args.ImageSize)
        else:
            if not Path(args.train_dir).exists():
                print(f"[ERR] Training set path does not exist (for calibration split): {args.train_dir}")
                sys.exit(1)
            _, calib_ds, _ = split_normal_dataset(args.train_dir, args.calib_ratio, args.seed, args.ImageSize)

        calib_scores, _, _, _ = anogan_scores_on_dataset(
            calib_ds, G, D, device, max_items=args.max_items,
            z_steps=args.z_steps, score_combo=args.score_combo
        )
        print(f"[INFO] Calibration samples: {len(calib_scores)}")

        results = []

        # 2. Loop through different SNR levels
        for snr in args.snr_levels:
            print(f"\n--- Testing SNR = {snr} dB ---")

            # 2.1 Inject noise and infer (Time consuming step)
            test_scores, test_labels, _, _ = anogan_scores_on_dataset(
                test_ds, G, D, device, max_items=args.max_items,
                z_steps=args.z_steps, z_lr=args.z_lr,
                lambda_feat=args.lambda_feat, score_combo=args.score_combo,
                snr_db=snr  # Inject noise here
            )

            # 2.2 Loop through different thresholds (Quantiles)
            for q in args.test_quantiles:
                # Dynamically calculate Tau
                tau = float(np.quantile(calib_scores, q))
                # Compute metrics
                m = compute_metrics_from_scores(test_scores, test_labels, tau)
                m['SNR_dB'] = snr
                m['Quantile'] = q
                results.append(m)

                if wrun:
                    wrun.log({f"snr_test/Acc_q{q}": m['Acc'], "snr": snr})

        # 3. Print Summary Table
        print("\n" + "=" * 90)
        print(f"{'SNR(dB)':<10} | {'Quantile':<10} | {'Tau':<10} | {'Acc':<10} | {'F1':<10} | {'AUROC':<10}")
        print("-" * 90)
        for r in results:
            print(f"{r['SNR_dB']:<10.1f} | {r['Quantile']:<10.3f} | {r['tau']:<10.4f} | "
                  f"{r['Acc']:<10.4f} | {r['F1']:<10.4f} | {r['AUROC']:<10.4f}")
        print("=" * 90 + "\n")

        # 4. Save CSV
        csv_path = Path(args.log_dir) / "snr_robustness_detailed.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        keys = results[0].keys()
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"[INFO] Detailed data saved to {csv_path}")

        # 5. Plot
        plt.figure(figsize=(10, 6))
        unique_quantiles = sorted(list(set(r['Quantile'] for r in results)))
        for q in unique_quantiles:
            subset = [r for r in results if r['Quantile'] == q]
            subset.sort(key=lambda x: x['SNR_dB'], reverse=True)
            x_snr = [r['SNR_dB'] for r in subset]
            y_acc = [r['Acc'] for r in subset]
            plt.plot(x_snr, y_acc, marker='o', label=f'Quantile={q}')

        plt.xlabel("SNR (dB)")
        plt.ylabel("Accuracy")
        plt.title("Robustness: Accuracy vs Noise Level")
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1.05)
        plt.savefig(Path(args.log_dir) / "snr_robustness_accuracy.png", dpi=150)

        if wrun: wrun.finish()
        return

    # ==========================================
    # === Mode: EVAL (Standard Test) ===
    # ==========================================
    if args.mode == "eval":
        # Determine Tau
        tau = None
        # Prioritize reading from file
        if args.tau_path and Path(args.tau_path).exists():
            try:
                tau = float(Path(args.tau_path).read_text().split("tau=")[-1].strip())
                print(f"[INFO] Loaded tau from file: {tau:.6f}")
            except Exception:
                pass

        # Otherwise recompute
        if tau is None:
            print("[INFO] Computing tau from train/calib split...")
            calib_ds = None
            if args.calib_dir:
                calib_ds = SpectrogramNPZDataset(args.calib_dir, classes=("Noleak",), resize_to=args.ImageSize)
            else:
                if not Path(args.train_dir).exists():
                    print(f"[ERR] Training set path does not exist: {args.train_dir}")
                    sys.exit(1)
                _, calib_ds, _ = split_normal_dataset(args.train_dir, args.calib_ratio, args.seed, args.ImageSize)

            c_scores, _, _, _ = anogan_scores_on_dataset(calib_ds, G, D, device, max_items=args.max_items,
                                                         z_steps=args.z_steps, score_combo=args.score_combo)
            tau = float(np.quantile(c_scores, args.quantile))
            print(f"[INFO] Computed Tau = {tau:.6f} (quantile={args.quantile})")

        # Score Test Set
        print("\n[INFO] Scoring Test Set (Clean)...")
        scores, labels, _, _ = anogan_scores_on_dataset(
            test_ds, G, D, device, max_items=args.max_items,
            z_steps=args.z_steps, z_lr=args.z_lr, lambda_feat=args.lambda_feat, score_combo=args.score_combo
        )

        metrics = compute_metrics_from_scores(scores, labels, tau)
        metrics["score_combo"] = args.score_combo
        print("\n[INFO] Metrics:")
        print(json.dumps(metrics, indent=2))
        if wrun: wrun.log({f"eval/{k}": v for k, v in metrics.items()})

        # Save metrics
        with open(Path(args.log_dir) / "eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        if wrun: wrun.finish()
        return

    # ==========================================
    # === Mode: TRAIN ===
    # ==========================================
    wrun = None
    if args.wandb:
        wrun = try_import_wandb(True)
        if wrun:
            wrun.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    single_run_train(args, wandb_run=wrun)

    if wrun: wrun.finish()


if __name__ == "__main__":
    main()