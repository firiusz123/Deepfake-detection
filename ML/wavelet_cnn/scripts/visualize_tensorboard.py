#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import config.config as cfg
from models.wavelet_model import WaveletHybridNet


class TensorBoardWaveletWrapper(nn.Module):
    """
    Wrapper to make TensorBoard tracing easier by avoiding dict inputs.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_rgb, ll1, lh1, hl1, hh1, ll2, lh2, hl2, hh2):
        wav = {
            "LL1": ll1,
            "LH1": lh1,
            "HL1": hl1,
            "HH1": hh1,
            "LL2": ll2,
            "LH2": lh2,
            "HL2": hl2,
            "HH2": hh2,
        }
        return self.model(x_rgb, wav)


def parse_args():
    parser = argparse.ArgumentParser(description="Export WaveletHybridNet graph to TensorBoard.")
    parser.add_argument("--logdir", type=str, default="runs/wavelet_graph", help="TensorBoard log directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size.")
    parser.add_argument("--img-size", type=int, default=cfg.IMG_SIZE, help="Dummy RGB input H/W.")
    parser.add_argument("--wav-size", type=int, default=64, help="Dummy wavelet map H/W.")
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        choices=["cpu", "cuda"],
        help="Device for dummy forward pass.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    model = WaveletHybridNet(num_classes=cfg.NUM_CLASSES).to(device).eval()
    wrapped = TensorBoardWaveletWrapper(model).to(device).eval()

    b = args.batch_size
    i = args.img_size
    w = args.wav_size

    x_rgb = torch.randn(b, 3, i, i, device=device)
    ll1 = torch.randn(b, 1, w, w, device=device)
    lh1 = torch.randn(b, 1, w, w, device=device)
    hl1 = torch.randn(b, 1, w, w, device=device)
    hh1 = torch.randn(b, 1, w, w, device=device)
    ll2 = torch.randn(b, 1, w, w, device=device)
    lh2 = torch.randn(b, 1, w, w, device=device)
    hl2 = torch.randn(b, 1, w, w, device=device)
    hh2 = torch.randn(b, 1, w, w, device=device)

    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)
    writer.add_graph(
        wrapped,
        (x_rgb, ll1, lh1, hl1, hh1, ll2, lh2, hl2, hh2),
    )
    writer.close()

    print(f"[INFO] TensorBoard graph written to: {args.logdir}")
    print("[INFO] Run: tensorboard --logdir runs")


if __name__ == "__main__":
    main()
