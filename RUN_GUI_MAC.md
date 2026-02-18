# Running poke.AI GUI on macOS (Intel + Apple Silicon)

This guide is the operational setup for running the GUI on macOS.

## 1) Prerequisites

You need:
- macOS (Intel or Apple Silicon)
- Python 3.10 (recommended for this project)
- A Game Boy Advance emulator + Pokemon Emerald ROM
- This repository cloned locally

Optional but recommended:
- A detector model at `object_detection/keras-retinanet/inference_graphs/map_detector.h5`

If the detector model is missing, the app starts in `stub mode` (no object detection).

## 2) Install system tools

Install Homebrew if needed, then:

```bash
brew install python@3.10
brew install tesseract
```

## 3) Create and activate virtual environment

From repository root:

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 4) Install Python dependencies

### Apple Silicon (M1/M2/M3)

Install TensorFlow for macOS first:

```bash
python -m pip install tensorflow-macos tensorflow-metal
```

Then install project dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ./object_detection/keras-retinanet
```

### Intel Mac

Install project dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ./object_detection/keras-retinanet
```

## 5) Grant permissions required by screen capture/input automation

In macOS Settings, grant your terminal app (Terminal/iTerm/Python host):
- `Privacy & Security > Screen Recording`
- `Privacy & Security > Accessibility`

Without these permissions, `mss`/`pyautogui` will fail or return empty captures.

## 6) Prepare emulator and ROM before starting GUI

1. Open your emulator.
2. Load the Pokemon Emerald ROM.
3. Keep emulator window visible (not minimized).
4. Set game viewport close to:
- `width=720`
- `height=480`

The GUI expects this capture geometry.

## 7) (Optional) Add detector model

Place detector at:

`object_detection/keras-retinanet/inference_graphs/map_detector.h5`

If absent, startup logs:

`[poke-ai] WARNING: detection model not found ... Running in stub mode.`

## 8) Run the GUI

From repository root:

```bash
python -m ai.gui
```

## 9) Start / stop behavior

- Click `Start` in GUI to begin updates.
- Stop using any of:
  - `Ctrl+C` in terminal
  - `Esc` in GUI window
  - Close window (`X`)

## 10) If emulator window matching fails

If template matching cannot find emulator window, app falls back to manual coordinates in:

`ai/gui.py`:

```python
self.game_window_size = {"top": 0, "left": 0, "width": 720, "height": 480}
```

Set `top` and `left` to your emulator window position.

## 11) Active-learning capture output

Captured data is written to:

- `data/active_learning/frames`
- `data/active_learning/preds`

Use this for labeling and retraining.

## 12) Common failures and direct fixes

- `ModuleNotFoundError: No module named 'keras_retinanet'`
  - Run: `python -m pip install -e ./object_detection/keras-retinanet`

- GUI launches but no screen updates
  - Verify Screen Recording + Accessibility permissions.
  - Ensure emulator is visible and frontmost.

- `No file or directory ... map_detector.h5`
  - Provide model file or continue in stub mode.
