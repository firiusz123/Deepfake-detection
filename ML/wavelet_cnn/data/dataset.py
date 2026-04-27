import os
import torch
import numpy as np
import cv2
import pywt
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F


class WaveletDeepfakeDataset(Dataset):
    """
    RGB + 2-level wavelet dataset (stable production version)
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        self.samples = []
        self.labels = []

        # =====================================================
        # LOAD FILE PATHS
        # =====================================================
        for label_name in ["real", "fake"]:
            class_dir = os.path.join(self.root_dir, label_name)

            if not os.path.exists(class_dir):
                continue

            label = 0 if label_name == "real" else 1

            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)

                if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    self.samples.append(path)
                    self.labels.append(label)

        print(f"[INFO] Loaded {len(self.samples)} images from {split}")

    # =====================================================
    # STABLE WAVELET TRANSFORM
    # =====================================================
    def dwt2(self, img):
        """
        Stable 2D DWT with normalization per level
        """

        img = img.astype(np.float32)

        # convert to grayscale if RGB
        if img.ndim == 3:
            img = np.mean(img, axis=2)

        # IMPORTANT: stabilize distribution per level
        img = (img - np.mean(img)) / (np.std(img) + 1e-6)

        LL, (LH, HL, HH) = pywt.dwt2(img, "db4")

        return LL, LH, HL, HH

    # =====================================================
    # TENSOR CONVERSION (NO ARTIFICIAL CHANNEL EXPANSION)
    # =====================================================
    def to_tensor(self, x):
        x = torch.tensor(x, dtype=torch.float32)

        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # [1, H, W]

        # =====================================================
        # FORCE FIXED SIZE (CRITICAL FIX)
        # =====================================================
        x = F.interpolate(
            x.unsqueeze(0),   # add batch dim
            size=(64, 64),    # fixed wavelet resolution
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return x


    # =====================================================
    # SAFE IMAGE LOADING
    # =====================================================
    def __getitem__(self, idx):

        path = self.samples[idx]
        label = self.labels[idx]

        # ---------------- LOAD IMAGE ----------------
        img = cv2.imread(path)

        # skip broken images safely
        while img is None:
            idx = (idx + 1) % len(self.samples)
            path = self.samples[idx]
            img = cv2.imread(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # =====================================================
        # RGB BRANCH (PIL FOR TORCHVISION COMPATIBILITY)
        # =====================================================
        pil_img = Image.fromarray(img)

        if self.transform:
            rgb = self.transform(pil_img)
        else:
            rgb = torch.tensor(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        # normalize AFTER PIL conversion for consistency
        img = img.astype(np.float32) / 255.0

        # =====================================================
        # LEVEL 1 WAVELET
        # =====================================================
        LL1, LH1, HL1, HH1 = self.dwt2(img)

        # =====================================================
        # LEVEL 2 WAVELET
        # =====================================================
        LL2, LH2, HL2, HH2 = self.dwt2(LL1)

        # =====================================================
        # PACKAGE OUTPUT
        # =====================================================
        wav = {
            "LL1": self.to_tensor(LL1),
            "LH1": self.to_tensor(LH1),
            "HL1": self.to_tensor(HL1),
            "HH1": self.to_tensor(HH1),

            "LL2": self.to_tensor(LL2),
            "LH2": self.to_tensor(LH2),
            "HL2": self.to_tensor(HL2),
            "HH2": self.to_tensor(HH2),
        }

        return rgb, wav, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.samples)
