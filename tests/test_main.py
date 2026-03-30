import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from main import (
    CLASSES,
    SPLITS,
    build_dataset,
    collect_split_files,
    extract_features,
    fft_features,
    hsv_features,
    wavelet_features,
)


def _dummy_image(value: int, size: int = 32) -> np.ndarray:
    """Create a solid-color RGB image for testing."""
    return np.full((size, size, 3), value, dtype=np.uint8)


def _make_dummy_dataset(root: Path):
    """Create a minimal dataset with real/fake splits for every partition."""
    dataset_root = root / "Data Set 1" / "Data Set 1"
    for split in SPLITS:
        for label in CLASSES:
            split_dir = dataset_root / split / label
            split_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(2):
                img_path = split_dir / f"{label}_{idx}.jpg"
                # Alternate colors so classes are distinguishable.
                color = 50 if label == "real" else 200
                cv2.imwrite(str(img_path), _dummy_image(color))


class FeatureTests(unittest.TestCase):
    def setUp(self):
        gray = np.linspace(0, 255, 32, dtype=np.uint8)
        self.gradient = np.stack([np.tile(gray, (32, 1))] * 3, axis=-1)

    def test_hsv_features_shape_and_distribution(self):
        feats = hsv_features(self.gradient)
        hist_len = 16 * 3
        stats_len = 12
        self.assertEqual(feats.shape[0], hist_len + stats_len)
        self.assertAlmostEqual(float(feats[:16].sum()), 1.0, places=5)

    def test_fft_features_non_negative(self):
        feats = fft_features(self.gradient)
        self.assertEqual(feats.shape[0], 16)
        self.assertTrue(np.all(feats >= 0))

    def test_wavelet_features_valid_stats(self):
        feats = wavelet_features(self.gradient)
        self.assertGreater(feats.size, 0)
        self.assertFalse(np.isnan(feats).any())

    def test_extract_features_combinations(self):
        hsv_feats = hsv_features(self.gradient)
        combined = extract_features(self.gradient, use_hsv=True, use_fft=False, use_wavelet=False)
        self.assertEqual(combined.shape, hsv_feats.shape)
        combo = extract_features(self.gradient, use_hsv=True, use_fft=True, use_wavelet=True)
        self.assertGreater(combo.shape[0], combined.shape[0])


class DatasetSmokeTests(unittest.TestCase):
    def test_build_dataset_limit_keeps_multiple_classes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            _make_dummy_dataset(base_path)
            split_files = collect_split_files(base_path)
            features, labels = build_dataset(split_files["train"], size=32, use_hsv=True, use_fft=False, use_wavelet=False, limit=1)
            self.assertGreaterEqual(features.shape[0], 2)
            self.assertSetEqual(set(labels.tolist()), {0, 1})
            self.assertGreater(features.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
