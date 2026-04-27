#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pywt
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATA_ROOT = "/home/firiusz/Downloads/deepfakedata/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train"
SAMPLE_SIZE = 200
WAVELET = "db4"

# =========================
# FILE SCAN
# =========================
def get_images(root):
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(r, f))
    return paths


# =========================
# WAVELET TRANSFORM (2 LEVEL)
# =========================
def compute_wavelet(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float16) / np.float16(255.0)

    LL1, (LH1, HL1, HH1) = pywt.dwt2(img, WAVELET)
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, WAVELET)

    return {
        "LL1": LL1, "LH1": LH1, "HL1": HL1, "HH1": HH1,
        "LL2": LL2, "LH2": LH2, "HL2": HL2, "HH2": HH2,
    }


# =========================
# ESTIMATION (ALL FP16)
# =========================
def estimate_dataset_size(files):
    sample_files = files[:min(SAMPLE_SIZE, len(files))]

    total_image = 0
    total_wavelet = 0

    for path in tqdm(sample_files, desc="Processing sample"):
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # =====================================================
        # ORIGINAL IMAGE (FP16 RGB)
        # =====================================================
        # 3 channels × 2 bytes (float16)
        img_size = h * w * 3 * 2
        total_image += img_size

        # =====================================================
        # WAVELET (FP16 grayscale)
        # =====================================================
        wav = compute_wavelet(img)

        wav_size = 0
        for k, v in wav.items():
            # already float16
            wav_size += v.astype(np.float16).nbytes

        total_wavelet += wav_size

    avg_img = total_image / len(sample_files)
    avg_wav = total_wavelet / len(sample_files)

    return avg_img, avg_wav


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("\nScanning dataset...")
    files = get_images(DATA_ROOT)

    print(f"Total images found: {len(files)}")

    avg_img, avg_wav = estimate_dataset_size(files)

    total_img_est = avg_img * len(files)
    total_wav_est = avg_wav * len(files)

    print("\n================ ESTIMATION (FULL FLOAT16 PIPELINE) ================\n")

    print(f"Original dataset (FP16 training representation):")
    print(f"  ≈ {total_img_est / (1024**3):.2f} GB")

    print(f"\nWavelet cache (2-level grayscale FP16):")
    print(f"  ≈ {total_wav_est / (1024**3):.2f} GB")

    print("\n================ FINAL ANALYSIS ================\n")

    ratio = total_wav_est / total_img_est

    print(f"Wavelet / Image ratio: {ratio:.2f}x")

    if ratio < 0.5:
        print("✔ EXCELLENT: Very efficient pipeline. Strongly recommended.")
    elif ratio < 1.0:
        print("✔ GOOD: Balanced storage and speed.")
    elif ratio < 1.5:
        print("⚠ ACCEPTABLE: Still usable but heavy.")
    else:
        print("❌ TOO HEAVY: reconsider wavelet strategy.")

    print("\n==================================================\n")
