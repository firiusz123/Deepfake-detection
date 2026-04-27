#!/usr/bin/env python3
import argparse
import re
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
        description="Visualize WaveletHybridNet as a block diagram (torchview + graphviz)."
    )
    parser.add_argument("--output", type=str, default="wavelet_model_blocks", help="Output file path without extension.")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Output image format.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size.")
    parser.add_argument("--img-size", type=int, default=cfg.IMG_SIZE, help="Dummy RGB input H/W.")
    parser.add_argument("--wav-size", type=int, default=64, help="Dummy wavelet map H/W.")
    parser.add_argument("--depth", type=int, default=3, help="Nested module depth for graph expansion.")
    parser.add_argument("--dark", action="store_true", help="Enable dark/night theme for the rendered graph.")
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        choices=["cpu", "cuda"],
        help="Device used for dummy forward pass.",
    )
    return parser.parse_args()


def apply_dark_theme(graphviz_graph):
    graphviz_graph.attr(bgcolor="#0F1117")
    graphviz_graph.attr(
        "node",
        style="filled,rounded",
        fillcolor="#1F2430",
        color="#C0CAF5",
        fontcolor="#E6EDF3",
    )
    graphviz_graph.attr(
        "edge",
        color="#A9B1D6",
        fontcolor="#A9B1D6",
    )
    _force_dark_body_styles(graphviz_graph)


def _upsert_attr(line, key, value):
    pattern = rf'{key}="[^"]*"|{key}=[^,\]]+'
    replacement = f'{key}="{value}"'
    if re.search(pattern, line):
        return re.sub(pattern, replacement, line)
    return line.replace("]", f', {replacement}]')


def _force_dark_body_styles(graphviz_graph):
    styled = []
    for line in graphviz_graph.body:
        line_out = line
        has_attrs = "[" in line_out and "]" in line_out

        if has_attrs and "->" in line_out:
            line_out = _upsert_attr(line_out, "color", "#E6EDF3")
            line_out = _upsert_attr(line_out, "fontcolor", "#E6EDF3")
            line_out = _upsert_attr(line_out, "penwidth", "1.8")
        elif has_attrs and "->" not in line_out:
            line_out = _upsert_attr(line_out, "style", "filled,rounded")
            line_out = _upsert_attr(line_out, "fillcolor", "#1F2430")
            line_out = _upsert_attr(line_out, "color", "#C0CAF5")
            line_out = _upsert_attr(line_out, "fontcolor", "#E6EDF3")

        styled.append(line_out)

    graphviz_graph.body = styled


def main():
    args = parse_args()

    try:
        from torchview import draw_graph
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: torchview. Install with: pip install torchview graphviz"
        ) from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    model = WaveletHybridNet(num_classes=cfg.NUM_CLASSES).to(device).eval()
    x_rgb, wav = build_dummy_inputs(args.batch_size, args.img_size, args.wav_size, device)

    graph = draw_graph(
        model,
        input_data=(x_rgb, wav),
        depth=args.depth,
        expand_nested=True,
        graph_name="WaveletHybridNet",
    )
    if args.dark:
        apply_dark_theme(graph.visual_graph)

    graph.visual_graph.render(args.output, format=args.format, cleanup=True)
    print(f"[INFO] Model graph saved: {args.output}.{args.format}")


if __name__ == "__main__":
    main()
