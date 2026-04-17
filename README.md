# DeepTrace GUI — Spring Boot

Brutalist Java web frontend for the Deepfake Detection CNN.
The Java app handles the web UI and file uploads; Python does all the ML work.

## Project Structure

```
deepfake-gui/                  ← this Spring Boot project (open in IntelliJ)
  pom.xml
  src/main/java/com/deepfake/
    DeepfakeApplication.java
    controller/DeepfakeController.java
    service/InferenceService.java
    model/InferenceResult.java
  src/main/resources/
    templates/index.html       ← the brutalist frontend
    application.properties     ← config (Python path, port, etc.)

../infer.py                    ← Python inference script (in repo root)
../model_best.pth              ← trained model weights (after training)
../ML/cnn_baseline/model.py    ← SimpleCNN definition
```

## Setup

### 1. Requirements

- **Java 17+**
- **Maven** (or use the IntelliJ Maven plugin)
- **Python** with: `torch torchvision pillow`

### 2. Train the model (if not done yet)

```bash
cd ..   # repo root
python train.py --archive_path ./archive --mode train
# → saves model_best.pth in the repo root
```

### 3. Open in IntelliJ

1. Open IntelliJ → **File → Open** → select the `deepfake-gui/` folder
2. IntelliJ will auto-detect the Maven project and download dependencies
3. Wait for indexing to finish

### 4. Configure paths (if needed)

Edit `src/main/resources/application.properties`:

```properties
# Absolute path to infer.py (if running from a custom working dir)
deepfake.infer-script=../infer.py

# Use "python3" on Linux/macOS if that's your command
deepfake.python-cmd=python
```

**IntelliJ tip:** Set the working directory for the run configuration to the
`deepfake-gui/` folder so that `../infer.py` resolves to the repo root.
Go to: Run → Edit Configurations → Working Directory → set to `$MODULE_WORKING_DIR$`

### 5. Run

In IntelliJ: right-click `DeepfakeApplication.java` → **Run**

Or from terminal:
```bash
cd deepfake-gui
mvn spring-boot:run
```

Then open: **http://localhost:8080**

## How It Works

```
Browser  ──(multipart POST /predict)──►  DeepfakeController.java
                                                │
                                         saves temp file
                                                │
                                         InferenceService.java
                                                │
                                    ProcessBuilder: python infer.py <path>
                                                │
                                          infer.py
                                           loads SimpleCNN
                                           runs inference
                                           prints JSON
                                                │
                                         parse JSON
                                                │
Browser  ◄──────────────(JSON response)─────────┘
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `model_best.pth not found` | Train the model first: `python train.py ...` |
| `python: command not found` | Set `deepfake.python-cmd=python3` in properties |
| `Module ML not found` | Make sure `infer.py` is run from the repo root |
| Port 8080 in use | Change `server.port=9090` in properties |
