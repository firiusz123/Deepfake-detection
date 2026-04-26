# DeepTrace Snip

A desktop screen-snip tool that detects deepfakes instantly using your trained CNN model.

## How to use

1. Run the app — a **D** icon appears in your system tray (bottom right)
2. **Double-click** the tray icon (or right-click → Snip & Analyse)
3. Your screen dims — **drag to select** any region containing a face
4. A result popup shows **REAL** or **FAKE** with confidence bars

Press **ESC** during selection to cancel.

## Setup in IntelliJ

1. Open IntelliJ → **File → Open** → select this `deeptrace-snip/` folder
2. Let Maven load
3. Maven panel → **Lifecycle** → **package** to build the JAR
4. Run `DeepTraceSnip` directly from IntelliJ

## First run — check your paths

Right-click tray icon → **Settings** and confirm:

- **Python command:** `python` (or `python3`)
- **infer.py path:** `C:/Users/Flippy/Downloads/Deepfake-detection-master/Deepfake-detection-master/infer.py`

These are saved automatically between sessions.

## Build a standalone JAR

```
mvn package
```

Creates `target/deeptrace-snip.jar` — double-click to run anywhere.
