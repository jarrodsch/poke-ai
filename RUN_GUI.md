# Running poke.AI GUI (Step-by-Step)

This guide is the operational setup for running the GUI on Windows.

## 1) Prerequisites

You need:
- Python 3.10+
- A Game Boy Advance emulator window (the project was built for VBA-rerecording)
- A Pokemon Emerald ROM
- This repository cloned locally

Optional but recommended:
- A trained detector model at `object_detection/keras-retinanet/inference_graphs/map_detector.h5`

If the detector model is missing, the app will still run in `stub mode` (no object detection).

## 2) Create and activate virtual environment

From repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3) Install Python dependencies

From repository root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .\object_detection\keras-retinanet
```

Notes:
- The `-e` install is required so imports like `keras_retinanet` resolve from this repo.
- If `pyautogui` fails on screenshot confidence, install OpenCV bindings:

```powershell
python -m pip install opencv-python
```

## 4) Prepare emulator and ROM before starting GUI

1. Open your emulator.
2. Load your Pokemon Emerald ROM.
3. Keep the emulator window visible on screen.
4. Set game viewport to match expected size:
- `width=720`
- `height=480`

The GUI captures this area for control and mapping.

## 5) (Optional) Add detector model

Place the detector file at:

`object_detection/keras-retinanet/inference_graphs/map_detector.h5`

If missing, startup prints:

`[poke-ai] WARNING: detection model not found ... Running in stub mode.`

That is expected and not fatal.

## 6) Run the GUI

From repository root:

```powershell
python -m ai.gui
```

## 7) Start / stop behavior

- Press the GUI `Start` button to begin updates.
- Stop methods:
  - `Ctrl+C` in terminal
  - `Esc` key in GUI window
  - Close GUI window (`X`)

## 8) If emulator auto-detection fails

If template matching cannot locate the emulator window, the app falls back to manual coordinates from:

`ai/gui.py`:

```python
self.game_window_size = {"top": 0, "left": 0, "width": 720, "height": 480}
```

Adjust `top` and `left` to match your screen position, then rerun.

## 9) Active-learning capture output

When running, capture output is written to:

- `data/active_learning/frames`
- `data/active_learning/preds`

These files are intended for later labeling/retraining loops.

## 10) Common failures and direct fixes

- `ModuleNotFoundError: No module named 'keras_retinanet'`
  - Run: `python -m pip install -e .\object_detection\keras-retinanet`

- `ImageNotFoundException` from `pyautogui` at startup
  - Ensure emulator window is visible.
  - Or set manual `top/left` in `ai/gui.py`.

- `No file or directory ... map_detector.h5`
  - Either provide the model file or run in stub mode.

- GUI starts but appears unresponsive
  - Click `Start` in GUI.
  - Confirm emulator is open and not minimized.
