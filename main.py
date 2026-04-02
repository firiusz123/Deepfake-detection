#!/usr/bin/env python3
import argparse
import itertools
import math
import os
import sys
from pathlib import Path

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
    roots = []
    for ds_dir in sorted(base_dir.glob("Data Set *")):
        if not ds_dir.is_dir():
            continue
        inner = ds_dir / ds_dir.name
        if inner.is_dir():
            roots.append(inner)
    return roots


def collect_split_files(base_dir: Path):
    roots = find_dataset_roots(base_dir)
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


def extract_features(img, use_hsv, use_fft, use_wavelet):
    feats = []
    if use_hsv:
        feats.append(hsv_features(img))
    if use_fft:
        feats.append(fft_features(img))
    if use_wavelet:
        feats.append(wavelet_features(img))
    if not feats:
        raise ValueError("No feature types selected")
    return np.concatenate(feats)


# -----------------------------
# Experiment runner
# -----------------------------

def build_dataset(split_files, size, use_hsv, use_fft, use_wavelet, limit=None):
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
        feats = extract_features(img, use_hsv, use_fft, use_wavelet)
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


def run_experiments(dataset_dir: Path, size: int, smoke: bool = False, output_csv: str = "results.csv"):
    split_files = collect_split_files(dataset_dir)

    # Permutations of HSV/FFT/WAVELET (7 non-empty combos)
    feature_names = ["HSV", "FFT", "WAVELET"]
    combos = []
    for r in range(1, 4):
        for combo in itertools.combinations(feature_names, r):
            combos.append(combo)

    results = []
    for combo in combos:
        use_hsv = "HSV" in combo
        use_fft = "FFT" in combo
        use_wavelet = "WAVELET" in combo

        limit = 20 if smoke else None

        X_train, y_train = build_dataset(split_files["train"], size, use_hsv, use_fft, use_wavelet, limit=limit)
        X_val, y_val = build_dataset(split_files["validation"], size, use_hsv, use_fft, use_wavelet, limit=limit)
        X_test, y_test = build_dataset(split_files["test"], size, use_hsv, use_fft, use_wavelet, limit=limit)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        clf = LinearSVC()
        clf.fit(X_train_scaled, y_train)

        val_metrics = evaluate_model(clf, scaler, X_val, y_val)
        test_metrics = evaluate_model(clf, scaler, X_test, y_test)

        results.append({
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
        })

        print(f"Done: {combo} | feat_len={X_train.shape[1]} | val_acc={val_metrics['accuracy']:.4f} | test_acc={test_metrics['accuracy']:.4f}")

    df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); output_path = output_csv.replace(".csv", f"_{timestamp}.csv"); df.to_csv(output_path, index=False); print(f"Saved results to {output_path}")

    # Rank by validation F1
    ranked = df.sort_values(by="val_f1", ascending=False)
    print("\nTop permutations by validation F1:")
    for _, row in ranked.iterrows():
        print(
            f"{row['features']}: val_f1={row['val_f1']:.4f}, val_acc={row['val_accuracy']:.4f}, "
            f"test_f1={row['test_f1']:.4f}, test_acc={row['test_accuracy']:.4f}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="HSV/FFT/Wavelet feature experiments")
    p.add_argument("--dataset-dir", type=str, default="archive", help="Path to dataset root (contains Data Set X)")
    p.add_argument("--size", type=int, default=256, help="Resize images to size x size")
    p.add_argument("--smoke", action="store_true", help="Run on 20 images per split for a smoke test")
    p.add_argument("--output-csv", type=str, default="results.csv", help="Where to write results CSV")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_experiments(Path(args.dataset_dir), size=args.size, smoke=args.smoke, output_csv=args.output_csv)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
