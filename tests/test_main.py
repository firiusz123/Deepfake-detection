import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from main import (
    CLASSES,
    SPLITS,
    brightness_contrast_features,
    build_dataset,
    collect_split_files,
    extract_features,
    fft_features,
    hsv_features,
    metadata_flags,
    noise_fingerprint,
    patch_consistency_features,
    run_experiments,
    wavelet_features,
)


def _dummy_image(value: int, size: int = 32) -> np.ndarray:
    """Create a solid-color RGB image for testing."""
    return np.full((size, size, 3), value, dtype=np.uint8)


def _make_dummy_dataset(root: Path, dataset_number: int = 1):
    """Create a minimal dataset with real/fake splits for every partition."""
    dataset_root = root / f"DataSet{dataset_number}"
    for split in SPLITS:
        for label in CLASSES:
            split_dir = dataset_root / split / label
            split_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(2):
                img_path = split_dir / f"{label}_{idx}.jpg"
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
        combined = extract_features(
            self.gradient, None, use_hsv=True, use_fft=False, use_wavelet=False
        )
        self.assertEqual(combined.shape, hsv_feats.shape)
        combo = extract_features(
            self.gradient, None, use_hsv=True, use_fft=True, use_wavelet=True
        )
        self.assertGreater(combo.shape[0], combined.shape[0])

    def test_extract_features_with_extra_components(self):
        base = extract_features(
            self.gradient, None, use_hsv=True, use_fft=False, use_wavelet=False
        )
        with_noise = extract_features(
            self.gradient,
            None,
            use_hsv=True,
            use_fft=False,
            use_wavelet=False,
            enable_noise_features=True,
            enable_patch_consistency=True,
        )
        self.assertGreater(with_noise.shape[0], base.shape[0])
        self.assertEqual(noise_fingerprint(self.gradient).shape[0], 9)
        self.assertEqual(patch_consistency_features(self.gradient).shape[0], 7)
        self.assertEqual(brightness_contrast_features(self.gradient).shape[0], 18)
        metadata = metadata_flags(Path("missing.jpg"), self.gradient)
        self.assertEqual(metadata.shape[0], 4)
        self.assertTrue(np.all(np.isfinite(metadata)))


class DatasetSmokeTests(unittest.TestCase):
    def test_build_dataset_limit_keeps_multiple_classes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            _make_dummy_dataset(base_path)
            split_files = collect_split_files(base_path)
            features, labels = build_dataset(
                split_files["train"],
                size=32,
                use_hsv=True,
                use_fft=False,
                use_wavelet=False,
                limit=1,
                enable_noise_features=True,
                enable_metadata_flags=True,
                enable_patch_consistency=True,
            )
            self.assertGreaterEqual(features.shape[0], 2)
            self.assertSetEqual(set(labels.tolist()), {0, 1})
        self.assertGreater(features.shape[1], 0)


class SelectionTests(unittest.TestCase):
    def test_dataset_selection_filters_datasets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            _make_dummy_dataset(base_path, dataset_number=1)
            _make_dummy_dataset(base_path, dataset_number=2)
            all_files = collect_split_files(base_path)
            filtered = collect_split_files(base_path, dataset_indices={1})
            self.assertEqual(len(all_files["train"]), 8)
            self.assertEqual(len(filtered["train"]), 4)
            self.assertTrue(len(filtered["validation"]) < len(all_files["validation"]))


class RegressionSmokeTests(unittest.TestCase):
    def test_run_experiments_outputs_overfit_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            _make_dummy_dataset(base_path)
            output_csv = Path(tmpdir) / "regression.csv"
            run_experiments(
                base_path,
                size=32,
                smoke=True,
                output_csv=str(output_csv),
                svm_kernels="linear",
                enable_noise_features=True,
                enable_metadata_flags=True,
                enable_patch_consistency=True,
                overfit_gap_threshold=0.0,
            )
            df = pd.read_csv(output_csv)
            self.assertIn("f1_gap", df.columns)
            self.assertIn("roc_gap", df.columns)
            self.assertIn("overfit_flag", df.columns)
            self.assertTrue(df["f1_gap"].ge(0).all())
            if not df["roc_gap"].isna().all():
                self.assertTrue(df["roc_gap"].ge(0).fillna(True).all())
            self.assertTrue(df["overfit_flag"].any())


if __name__ == "__main__":
    unittest.main()
