# DeepTrace — AI Deepfake Detection

> Detect AI-generated and deepfake images instantly using a CNN-based image forensics system.

---

## ⬇️ Download the App (No Install Required)

**[👉 Click here to go to the latest release](../../releases/latest)**

Download `DeepTrace-Ready.zip` from the release assets, extract it anywhere, and run `DeepTrace.exe`.

That's it. No Java, no Python, no installation needed.

---

## What's Inside

```
DeepTrace/
  DeepTrace.exe     ← the desktop snip app
  infer.exe         ← the AI inference engine (Python + model bundled)
  model_best.pth    ← trained CNN model weights
  ML/               ← model architecture files
  app/              ← Java runtime
  runtime/          ← bundled JRE
```

---

## How to Use DeepTrace Snip

1. **Run** `DeepTrace.exe` — a yellow **D** icon appears in your system tray (bottom-right taskbar)
2. **Double-click** the tray icon to start snipping
3. Your screen dims — **drag** to draw a box around any face or image
4. Release — the AI analyses it and shows **REAL** or **FAKE** with a confidence score
5. Press **ESC** or **right-click** to cancel at any time

### First Time Setup

If you get a path error, right-click the tray icon → **Settings** and set the infer path to:
```
C:\path\to\where\you\extracted\DeepTrace\infer.exe
```

---

## Web Interface (Developers)

If you want the browser-based drag-and-drop GUI instead:

### Requirements
- Java 17+
- Python 3.x with: `torch torchvision pillow pywavelets scikit-learn numpy`
- Maven (or use IntelliJ)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/firiusz123/Deepfake-detection.git
cd Deepfake-detection

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Place model weights in the repo root
# Download best_fold_0.pt from the latest release and put it here

# 4. Run the Spring Boot web app
cd java-gui
mvn spring-boot:run

# 5. Open in browser
# http://localhost:8080
```

---

## How It Works

```
Image Upload
    ↓
Spring Boot (Java 17) — localhost:8080
    ↓
infer_v2.py — WaveletHybridNet
    ↓
RGB Branch + Wavelet Branch (db4, 2 levels)
    ↓
Attention Fusion
    ↓
REAL / FAKE + Confidence %
```

The model uses a **WaveletHybridNet** architecture that processes images through two parallel branches — a standard RGB CNN and a wavelet frequency decomposition branch that detects the subtle high-frequency artifacts left behind by GAN and diffusion model generators.

---

## Project Structure

```
Deepfake-detection/
  ML/
    cnn_baseline/       ← v1 SimpleCNN model
    wavelet_cnn/        ← v2 WaveletHybridNet model (current)
  java-gui/             ← Spring Boot web GUI
  infer.py              ← v1 inference script
  infer_v2.py           ← v2 inference script (WaveletHybridNet)
  train.py              ← training script
  requirements.txt      ← Python dependencies
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | PyTorch |
| Model | WaveletHybridNet (RGB + Wavelet branches) |
| Web Backend | Spring Boot 3.2 / Java 17 |
| Templating | Thymeleaf |
| Desktop App | Java Swing + System Tray |
| Python Packaging | PyInstaller |
| Java Packaging | jpackage |

---

## Branch

Active development is on `feature/java-gui`.

---

## License

**Copyright © 2026 firiusz123. All Rights Reserved.**

This software and its source code are the exclusive property of the author.
No part of this project — including but not limited to the source code, model weights,
trained data, documentation, or compiled executables — may be reproduced, distributed,
modified, or used in any form without the express written permission of the author.

---

## ⚠️ Windows Security Note

When running `DeepTrace.exe` or `infer.exe` for the first time, Windows may show a security warning because the files are unsigned. To allow them:

1. Right-click the file → **Properties**
2. At the bottom tick **Unblock** → **OK**

Or add the folder to Windows Defender exclusions:
**Windows Security → Virus & threat protection → Exclusions → Add folder**
