#!/usr/bin/env python3
"""
infer.py — Single-image inference script for DeepTrace GUI.

Called by the Spring Boot backend as a subprocess:
    python infer.py <image_path>

Outputs a single line of JSON to stdout:
    {"label":"FAKE","confidence":94.7,"realProb":5.3,"fakeProb":94.7}

On error:
    {"label":null,"confidence":0,"realProb":0,"fakeProb":0,"error":"<message>"}

Place this file in the same directory as model_best.pth (the repo root).
"""

import sys
import os
import json

# ── Locate model & ML package ───────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MODEL_PATH = os.path.join(SCRIPT_DIR, "model_best.pth")
IMG_SIZE   = 256


def _error(msg: str) -> None:
    print(json.dumps({
        "label": None,
        "confidence": 0,
        "realProb": 0,
        "fakeProb": 0,
        "error": msg
    }))
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        _error("Usage: python infer.py <image_path>")

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        _error(f"File not found: {image_path}")

    # ── Lazy imports (keep startup fast for error cases) ─────────────────
    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
        from ML.cnn_baseline.model import SimpleCNN
    except ImportError as e:
        _error(f"Missing dependency: {e}. Run: pip install -r requirements.txt")

    # ── Load model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SimpleCNN(num_classes=2, img_size=IMG_SIZE)

    if not os.path.exists(MODEL_PATH):
        _error(f"model_best.pth not found at {MODEL_PATH}. Train the model first.")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # ── Preprocess (must match training transforms in dataset.py) ─────────
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    try:
        img    = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        _error(f"Could not open image: {e}")

    # ── Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    real_prob = round(probs[0].item() * 100, 4)
    fake_prob = round(probs[1].item() * 100, 4)
    label     = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = round(max(real_prob, fake_prob), 2)

    print(json.dumps({
        "label":      label,
        "confidence": confidence,
        "realProb":   real_prob,
        "fakeProb":   fake_prob,
        "error":      None
    }))


if __name__ == "__main__":
    main()
