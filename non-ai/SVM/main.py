#!/usr/bin/env python3
import argparse
import datetime
import itertools
import math
import os
import re
import sys
from pathlib import Path
from typing import Optional, Set

import cv2
import numpy as np
import pandas as pd
import pywt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# -----------------------------
# Data loading
# -----------------------------
SPLITS = ["train", "validation", "test"]
CLASSES = ["real", "fake"]


def find_dataset_roots(base_dir: Path):
    def _has_all_splits(path: Path) -> bool:
        return path.is_dir() and all((path / split).is_dir() for split in SPLITS)

    if not base_dir.is_dir():
        return []

    if _has_all_splits(base_dir):
        return [base_dir]

    roots = []
    for ds_dir in sorted(base_dir.iterdir()):
        if _has_all_splits(ds_dir):
            roots.append(ds_dir)
    return roots


def _extract_dataset_index(name: str) -> Optional[int]:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else None


def collect_split_files(base_dir: Path, dataset_indices: Set[int] | None = None):
    roots = find_dataset_roots(base_dir)
    if dataset_indices:
        filtered = []
        for root in roots:
            idx = _extract_dataset_index(root.name)
            if idx is not None and idx in dataset_indices:
                filtered.append(root)
        roots = filtered

    if not roots:
        raise FileNotFoundError(f"No dataset roots found in: {base_dir}")

    split_files = {split: [] for split in SPLITS}
    for root in roots:
        for split in SPLITS:
            for label in CLASSES:
                p = root / split / label
                if not p.is_dir():
                    continue
                for img in p.glob("*.jpg"):
                    split_files[split].append((img, 1 if label == "fake" else 0))
                for img in p.glob("*.png"):
                    split_files[split].append((img, 1 if label == "fake" else 0))
    return split_files


# -----------------------------
# Preprocessing
# -----------------------------

def load_image(path: Path, size: int):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


# -----------------------------
# Feature extraction
# -----------------------------

def hsv_features(img, h_bins=16, s_bins=16, v_bins=16):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    h_hist = cv2.calcHist([h], [0], None, [h_bins], [0, 180]).flatten()
    s_hist = cv2.calcHist([s], [0], None, [s_bins], [0, 256]).flatten()
    v_hist = cv2.calcHist([v], [0], None, [v_bins], [0, 256]).flatten()

    # Normalize histograms
    h_hist = h_hist / (h_hist.sum() + 1e-8)
    s_hist = s_hist / (s_hist.sum() + 1e-8)
    v_hist = v_hist / (v_hist.sum() + 1e-8)

    # Summary stats per channel
    stats = []
    for ch in (h, s, v):
        ch = ch.astype(np.float32)
        stats.extend([float(ch.mean()), float(ch.std()), float(ch.min()), float(ch.max())])

    return np.concatenate([h_hist, s_hist, v_hist, np.array(stats, dtype=np.float32)])


def fft_features(img, radial_bins=16):
    # Use grayscale for FFT
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    mag = np.log1p(magnitude)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = r.max()

    bins = np.linspace(0, r_max, radial_bins + 1)
    feats = []
    for i in range(radial_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.any():
            feats.append(float(mag[mask].mean()))
        else:
            feats.append(0.0)
    return np.array(feats, dtype=np.float32)


def wavelet_features(img, wavelet="db2", level=2):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=level)

    feats = []
    # coeffs[0] is approximation; coeffs[1:] are detail tuples (cH, cV, cD)
    for i, c in enumerate(coeffs):
        if i == 0:
            subbands = [c]
        else:
            subbands = list(c)
        for sb in subbands:
            sb = np.asarray(sb)
            feats.extend([
                float(sb.mean()),
                float(sb.std()),
                float(sb.min()),
                float(sb.max()),
            ])
    return np.array(feats, dtype=np.float32)


def brightness_contrast_features(img):
    img = img.astype(np.float32)
    feats = []
    for channel_idx in range(img.shape[2]):
        channel = img[..., channel_idx]
        feats.extend([
            float(channel.mean()),
            float(channel.std()),
            float(channel.min()),
            float(channel.max()),
            float(np.median(channel)),
            float(np.percentile(channel, 75.0)),
        ])
    return np.array(feats, dtype=np.float32)


def noise_fingerprint(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    filters = [
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        cv2.Laplacian(gray, cv2.CV_32F, ksize=3),
    ]
    feats = []
    for arr in filters:
        feats.extend([
            float(arr.mean()),
            float(arr.std()),
            float(np.percentile(arr, 75.0)),
        ])
    return np.array(feats, dtype=np.float32)


def patch_consistency_features(img, patch_size=16):
    h, w = img.shape[:2]
    patch_size = max(4, min(patch_size, h, w))
    centers = [
        (0, 0),
        (0, w - patch_size),
        (h - patch_size, 0),
    ]
    feats = []
    for row, col in centers:
        patch = img[row:row + patch_size, col:col + patch_size]
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
        feats.append(float(gray.mean()))
        feats.append(float(gray.std()))
    if feats:
        feats.append(float(max(feats) - min(feats)))
    else:
        feats.append(0.0)
    return np.array(feats, dtype=np.float32)


def metadata_flags(path: Path, img):
    h, w = img.shape[:2]
    exists = float(path.exists())
    avg = float(np.mean(img))
    return np.array([exists, float(h), float(w), avg], dtype=np.float32)


def extract_features(
    img,
    metadata: Optional[Path] = None,
    use_hsv=True,
    use_fft=True,
    use_wavelet=True,
    enable_noise_features=False,
    enable_patch_consistency=False,
    enable_metadata_flags=False,
):
    feats = []
    if use_hsv:
        feats.append(hsv_features(img))
    if use_fft:
        feats.append(fft_features(img))
    if use_wavelet:
        feats.append(wavelet_features(img))
    if enable_noise_features:
        feats.append(noise_fingerprint(img))
    if enable_patch_consistency:
        feats.append(patch_consistency_features(img))
    if enable_metadata_flags:
        meta_path = metadata if metadata is not None else Path(".")
        feats.append(metadata_flags(meta_path, img))
    if not feats:
        raise ValueError("No feature types selected")
    return np.concatenate(feats)


# -----------------------------
# Experiment runner
# -----------------------------

def build_dataset(
    split_files,
    size,
    use_hsv,
    use_fft,
    use_wavelet,
    limit=None,
    enable_noise_features=False,
    enable_metadata_flags=False,
    enable_patch_consistency=False,
):
    entries = list(split_files)
    available_classes = {label for _, label in entries}

    X = []
    y = []
    count = 0
    seen_classes = set()

    for path, label in entries:
        img = load_image(path, size)
        if img is None:
            continue
        feats = extract_features(
            img,
            path,
            use_hsv=use_hsv,
            use_fft=use_fft,
            use_wavelet=use_wavelet,
            enable_noise_features=enable_noise_features,
            enable_metadata_flags=enable_metadata_flags,
            enable_patch_consistency=enable_patch_consistency,
        )
        X.append(feats)
        y.append(label)
        seen_classes.add(label)
        count += 1
        if limit is not None and count >= limit and seen_classes >= available_classes:
            break

    if not X:
        raise RuntimeError("No images were loaded for this split.")
    return np.vstack(X), np.array(y, dtype=np.int32)


def evaluate_model(clf, scaler, X, y):
    Xs = scaler.transform(X)
    preds = clf.predict(Xs)

    # LinearSVC does not provide probabilities; use decision function for AUC
    scores = clf.decision_function(Xs)

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
    }

    # ROC-AUC requires both classes present
    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = roc_auc_score(y, scores)
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def run_experiments(
    dataset_dir: Path,
    size: int,
    smoke: bool = False,
    output_csv: str = "results.csv",
    svm_kernels: str = "linear",
    enable_noise_features: bool = False,
    enable_metadata_flags: bool = False,
    enable_patch_consistency: bool = False,
    overfit_gap_threshold: float = 0.0,
):
    split_files = collect_split_files(dataset_dir)

    feature_names = ["HSV", "FFT", "WAVELET"]
    combos = []
    for r in range(1, 4):
        for combo in itertools.combinations(feature_names, r):
            combos.append(combo)

    kernels = [kernel.strip() for kernel in svm_kernels.split(",") if kernel.strip()]
    if not kernels:
        kernels = ["linear"]

    limit = 20 if smoke else None
    results = []

    def _gap(a, b):
        if math.isnan(a) or math.isnan(b):
            return float("nan")
        return abs(a - b)

    for combo in combos:
        use_hsv = "HSV" in combo
        use_fft = "FFT" in combo
        use_wavelet = "WAVELET" in combo

        X_train, y_train = build_dataset(
            split_files["train"],
            size,
            use_hsv,
            use_fft,
            use_wavelet,
            limit=limit,
            enable_noise_features=enable_noise_features,
            enable_metadata_flags=enable_metadata_flags,
            enable_patch_consistency=enable_patch_consistency,
        )
        X_val, y_val = build_dataset(
            split_files["validation"],
            size,
            use_hsv,
            use_fft,
            use_wavelet,
            limit=limit,
            enable_noise_features=enable_noise_features,
            enable_metadata_flags=enable_metadata_flags,
            enable_patch_consistency=enable_patch_consistency,
        )
        X_test, y_test = build_dataset(
            split_files["test"],
            size,
            use_hsv,
            use_fft,
            use_wavelet,
            limit=limit,
            enable_noise_features=enable_noise_features,
            enable_metadata_flags=enable_metadata_flags,
            enable_patch_consistency=enable_patch_consistency,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        for kernel in kernels:
            clf = LinearSVC()
            clf.fit(X_train_scaled, y_train)

            val_metrics = evaluate_model(clf, scaler, X_val, y_val)
            test_metrics = evaluate_model(clf, scaler, X_test, y_test)

            f1_gap = _gap(val_metrics["f1"], test_metrics["f1"])
            roc_gap = _gap(val_metrics["roc_auc"], test_metrics["roc_auc"])
            overfit_flag = any(
                gap >= overfit_gap_threshold for gap in (f1_gap, roc_gap) if not math.isnan(gap)
            )

            results.append({
                "svm_kernel": kernel,
                "features": "+".join(combo),
                "feature_len": X_train.shape[1],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "f1_gap": f1_gap,
                "roc_gap": roc_gap,
                "overfit_flag": overfit_flag,
            })

            print(
                f"Done: kernel={kernel} | combo={combo} | feat_len={X_train.shape[1]} "
                f"| val_acc={val_metrics['accuracy']:.4f} | test_acc={test_metrics['accuracy']:.4f}"
            )

    df = pd.DataFrame(results)
    target_path = Path(output_csv)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = target_path.with_name(f"{target_path.stem}_{timestamp}{target_path.suffix}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(timestamped_path, index=False)
    df.to_csv(target_path, index=False)
    print(f"Saved results to {timestamped_path} and {target_path}")

    ranked = df.sort_values(by="val_f1", ascending=False)
    print("\nTop permutations by validation F1:")
    for _, row in ranked.iterrows():
        print(
            f"{row['features']}: val_f1={row['val_f1']:.4f}, val_acc={row['val_accuracy']:.4f}, "
            f"test_f1={row['test_f1']:.4f}, test_acc={row['test_accuracy']:.4f}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="HSV/FFT/Wavelet feature experiments")
    p.add_argument("--dataset-dir", type=str, default="archive", help="Path to dataset root (contains named DataSet folders)")
    p.add_argument("--size", type=int, default=256, help="Resize images to size x size")
    p.add_argument("--smoke", action="store_true", help="Run on 20 images per split for a smoke test")
    p.add_argument("--output-csv", type=str, default="results.csv", help="Where to write results CSV")
    p.add_argument("--svm-kernels", type=str, default="linear", help="Comma-separated kernel names (linear only)")
    p.add_argument("--enable-noise-features", action="store_true", help="Include noise fingerprint features in vectors")
    p.add_argument("--enable-metadata-flags", action="store_true", help="Append metadata-based flags per sample")
    p.add_argument("--enable-patch-consistency", action="store_true", help="Append patch consistency statistics")
    p.add_argument("--overfit-gap-threshold", type=float, default=0.0, help="Threshold to mark overfit gaps")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_experiments(
            Path(args.dataset_dir),
            size=args.size,
            smoke=args.smoke,
            output_csv=args.output_csv,
            svm_kernels=args.svm_kernels,
            enable_noise_features=args.enable_noise_features,
            enable_metadata_flags=args.enable_metadata_flags,
            enable_patch_consistency=args.enable_patch_consistency,
            overfit_gap_threshold=args.overfit_gap_threshold,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
