#!/usr/bin/env python3
import argparse
import torch

import config.config as cfg
from models.wavelet_model import WaveletHybridNet


def build_dummy_inputs(batch_size, img_size, wav_size, device):
    x_rgb = torch.randn(batch_size, 3, img_size, img_size, device=device)
    wav = {
        "LL1": torch.randn(batch_size, 1, wav_size, wav_size, device=device),
        "LH1": torch.randn(batch_size, 1, wav_size, wav_size, device=device),
        "HL1": torch.randn(batch_size, 1, wav_size, wav_size, device=device),
        "HH1": torch.randn(batch_size, 1, wav_size, wav_size, device=device),
        "LL2": torch.randn(batch_size, 1, wav_size, wav_size, device=device),
        "LH2": torch.randn(batch_size, 1, wav_size, wav_size, device=device),
        "HL2": torch.randn(batch_size, 1, wav_size, wav_size, device=device),
        "HH2": torch.randn(batch_size, 1, wav_size, wav_size, device=device),
    }
    return x_rgb, wav


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize WaveletHybridNet with hiddenlayer."
    )
    parser.add_argument("--output", type=str, default="wavelet_hiddenlayer", help="Output path without extension.")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Output format.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size.")
    parser.add_argument("--img-size", type=int, default=cfg.IMG_SIZE, help="Dummy RGB H/W.")
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

    try:
        import hiddenlayer as hl
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: hiddenlayer. Install with: pip install hiddenlayer graphviz"
        ) from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    model = WaveletHybridNet(num_classes=cfg.NUM_CLASSES).to(device).eval()
    x_rgb, wav = build_dummy_inputs(args.batch_size, args.img_size, args.wav_size, device)

    graph = hl.build_graph(model, (x_rgb, wav))
    graph.theme = hl.graph.THEMES["blue"].copy()
    graph.save(args.output, format=args.format)
    print(f"[INFO] HiddenLayer graph saved: {args.output}.{args.format}")


if __name__ == "__main__":
    main()
