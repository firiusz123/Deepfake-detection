#!/usr/bin/env python3

from data.dataset import WaveletDeepfakeDataset
import config.config as cfg


ds = WaveletDeepfakeDataset(
    root_dir=cfg.DATA_ROOT,
    split="train"
)

print(len(ds))

rgb, wav, label = ds[0]

print(rgb.shape)
print(label)
print(wav.keys())
