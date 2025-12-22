import argparse
from pathlib import Path
import numpy as np
from scipy.signal import stft, get_window, detrend, resample_poly
from scipy.io import wavfile
from typing import Optional, List, Tuple


# ==========================================
# Audio Preprocessing Utilities
# ==========================================

def read_wav_mono_float(path: Path) -> Tuple[int, np.ndarray]:
    """
    Read a WAV file and convert to mono float32 [-1, 1].
    """
    try:
        fs, x = wavfile.read(str(path))
    except ValueError as e:
        raise ValueError(f"Failed to read WAV file: {e}")

    # Convert to mono if stereo
    if x.ndim > 1:
        x = x[:, 0]

    # Normalize to float32 [-1, 1]
    if x.dtype not in (np.float32, np.float64):
        # Integer types (int16, int32, etc.)
        max_val = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / max_val
    else:
        x = x.astype(np.float32, copy=False)

    return fs, x


def maybe_resample(x: np.ndarray, fs: int, target_fs: Optional[int]) -> Tuple[int, np.ndarray]:
    """
    Resample audio to target_fs if necessary.
    """
    if target_fs is None or target_fs == fs or target_fs <= 0:
        return fs, x

    # Use GCD to find integer up/down ratios for polyphase resampling
    from math import gcd
    g = gcd(fs, target_fs)
    up, down = target_fs // g, fs // g

    x_rs = resample_poly(x, up, down)
    return target_fs, x_rs.astype(np.float32, copy=False)


def segment_signal(x: np.ndarray, fs: int, seg_sec: Optional[float]) -> List[np.ndarray]:
    """
    Split signal into fixed-length segments (discarding the tail).
    """
    if seg_sec is None or seg_sec <= 0:
        return [x]

    seg_len = int(round(seg_sec * fs))
    if seg_len > len(x):
        return []  # Signal is shorter than one segment

    n_full = len(x) // seg_len
    return [x[i * seg_len:(i + 1) * seg_len] for i in range(n_full)]


def compute_stft_db(x: np.ndarray, fs: int,
                    nperseg: int, noverlap: int, nfft: int,
                    window: str = "hann", detrend_on: bool = True, eps: float = 1e-12):
    """
    Compute STFT and convert magnitude to Decibels (dB).
    Returns: (frequencies, times, S_db[float32])
    """
    x_proc = detrend(x, type="linear") if detrend_on else x

    # Get window
    win = get_window(window, nperseg, fftbins=True)

    f, t, Zxx = stft(
        x_proc, fs=fs, window=win,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        boundary=None, padded=False
    )

    # Magnitude to dB
    S_db = 20.0 * np.log10(np.maximum(np.abs(Zxx), eps)).astype(np.float32, copy=False)

    return f.astype(np.float32), t.astype(np.float32), S_db


def process_folder(
        root_dir: str,
        resample_to: int,
        segment_sec: float,
        nperseg: int,
        noverlap: int,
        nfft: int,
        window: str,
        output_dir: Optional[str] = None
):
    """
    Recursively process all WAV files in a directory.
    Saves outputs as .npz files.
    """
    root = Path(root_dir)
    if not root.exists():
        print(f"[ERROR] Input directory not found: {root}")
        return

    wav_files = list(root.rglob("*.wav"))
    if not wav_files:
        print(f"[WARN] No WAV files found in: {root}")
        return

    print(f"[INFO] Found {len(wav_files)} WAV files.")
    print(f"[CONFIG] FS={resample_to}, Seg={segment_sec}s, STFT=({nperseg}/{noverlap}/{nfft})")

    success_count = 0

    for idx, wav_path in enumerate(sorted(wav_files)):
        try:
            # 1. Read and Resample
            fs_raw, x = read_wav_mono_float(wav_path)
            fs, x = maybe_resample(x, fs_raw, resample_to)

            # 2. Segment
            segs = segment_signal(x, fs, segment_sec)
            if not segs:
                print(f"[SKIP] {wav_path.name}: too short for {segment_sec}s segment.")
                continue

            # Determine output directory
            # If output_dir is provided, mirror the directory structure relative to root_dir
            # Otherwise, save in the same folder as the wav file
            if output_dir:
                rel_path = wav_path.parent.relative_to(root)
                target_folder = Path(output_dir) / rel_path
                target_folder.mkdir(parents=True, exist_ok=True)
            else:
                target_folder = wav_path.parent

            # 3. Compute STFT and Save
            if segment_sec is None or segment_sec <= 0:
                # Full file mode
                f, t, S_db = compute_stft_db(segs[0], fs, nperseg, noverlap, nfft, window)
                out_path = target_folder / f"{wav_path.stem}_stft.npz"
                np.savez_compressed(
                    out_path,
                    S_db=S_db,
                    f=f, t=t, fs=np.float32(fs),
                    nperseg=np.int32(nperseg), noverlap=np.int32(noverlap),
                    nfft=np.int32(nfft), window=np.string_(window),
                    seg_index=np.int32(-1)
                )
            else:
                # Segment mode
                for si, seg in enumerate(segs):
                    f, t, S_db = compute_stft_db(seg, fs, nperseg, noverlap, nfft, window)
                    out_path = target_folder / f"{wav_path.stem}_stft_seg{si:03d}.npz"
                    np.savez_compressed(
                        out_path,
                        S_db=S_db,
                        f=f, t=t, fs=np.float32(fs),
                        nperseg=np.int32(nperseg), noverlap=np.int32(noverlap),
                        nfft=np.int32(nfft), window=np.string_(window),
                        seg_index=np.int32(si),
                        seg_sec=np.float32(segment_sec)
                    )

            success_count += 1
            if (idx + 1) % 10 == 0:
                print(f"[{idx + 1}/{len(wav_files)}] Processed...")

        except Exception as e:
            print(f"[ERROR] Failed to process {wav_path}: {e}")

    print(f"[DONE] Successfully processed {success_count}/{len(wav_files)} files.")


# ==========================================
# Main / CLI
# ==========================================

if __name__ == "__main__":
    # my_dataset /
    # ├── raw_wav / < -- Use this path for --input_dir
    # │   │   ├── Noleak / < -- Store     Normal / Safe   recordings    here
    # │   │   ├── recording_001.wav
    # │   │   ├── recording_002.wav
    # │   │   └── ...
    # │   └── Leak / < -- Store Anomalous / Leak recordings here
    # │       ├── leak_sample_01.wav
    # │       ├── leak_sample_02.wav
    # │       └── ...
    parser = argparse.ArgumentParser(
        description="WAV to STFT Spectrogram Preprocessor (NPZ format)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Path Arguments
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing WAV files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as input)")

    # Signal Processing Arguments
    parser.add_argument("--fs", type=int, default=4096, help="Target sampling rate (Hz)")
    parser.add_argument("--seg_sec", type=float, default=10.0, help="Segment length in seconds (0 for full file)")

    # STFT Arguments
    parser.add_argument("--nperseg", type=int, default=1024, help="Length of each segment")
    parser.add_argument("--noverlap", type=int, default=512, help="Number of points to overlap")
    parser.add_argument("--nfft", type=int, default=1024, help="Length of the FFT")
    parser.add_argument("--window", type=str, default="hann", help="Window function type")

    args = parser.parse_args()

    print("=== Audio Preprocessing Started ===")
    process_folder(
        root_dir=args.input_dir,
        output_dir=args.output_dir,
        resample_to=args.fs,
        segment_sec=args.seg_sec,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        nfft=args.nfft,
        window=args.window
    )