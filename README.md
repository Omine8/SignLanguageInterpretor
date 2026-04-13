# SignBridge — Full Stack Sign Language Interpreter
## Complete Setup & Run Guide

---

## Folder Structure

```
I:\SIGNLANGUAGE\
├── run.py                  ← START HERE
├── requirements.txt
├── collect_data.py         ← your existing script
├── train_model.py          ← your existing script
├── sen_form_FINAL.py       ← your existing script
├── data_v2\                ← training data
├── models\                 ← saved models
└── backend\
    └── app.py              ← Flask server
└── frontend\
    └── templates\
        └── index.html      ← web dashboard
```

---

## Step 1 — Install dependencies

```powershell
# Activate your existing venv
i:\SIGNLANGUAGE\venv\Scripts\Activate.ps1

# Install Flask and CORS (everything else already installed)
pip install flask flask-cors google-generativeai==0.7.2
```

---

## Step 2 — Copy files into your project

Copy these files into `I:\SIGNLANGUAGE\`:
- `run.py`
- `requirements.txt`

Create these folders and copy files:
- `I:\SIGNLANGUAGE\backend\app.py`
- `I:\SIGNLANGUAGE\frontend\templates\index.html`

---

## Step 3 — Run the app

```powershell
cd I:\SIGNLANGUAGE
python run.py
```

Browser opens automatically at **http://localhost:5000**

---

## How to use the web dashboard

### Start detection
1. Click **Start Detection** in the sidebar
2. The webcam feed appears in the center
3. Show your hand — detected gestures appear in real time
4. Confidence bars update live on the right panel

### Sentence output
- Words accumulate in the **Sentence** area at the bottom of the video
- Click **Speak Output** to hear it spoken aloud via Google TTS
- Click **Undo Last Word** to remove the last word
- Click **Clear Sentence** to start over

### Collect data
1. Select a specific gesture from the dropdown (or leave blank for all)
2. Click **Collect Data**
3. OpenCV window opens — follow on-screen prompts
4. Logs stream in real time in the Logs tab

### Train model
1. Click **Train Model** (takes 20–30 min on CPU)
2. Watch training progress in the **Logs** tab
3. Model auto-reloads when training finishes

### Gemini LLM grammar
- Your API key is pre-filled
- Click **Connect Gemini** to verify it's working
- When connected, grammar correction upgrades from rule-based to LLM

---

## Fixing common issues

### "MediaPipe has no attribute solutions"
```powershell
pip install mediapipe==0.10.9 --force-reinstall
```

### "Gemini model not found"
```powershell
pip install google-generativeai==0.7.2 --force-reinstall
```

### "Flask not found"
```powershell
pip install flask flask-cors
```

### Camera doesn't open
- Make sure no other app is using the webcam
- Close any existing `sen_form_FINAL.py` windows first
- The web app controls the camera — don't run both at once

### Model mismatch error (9 classes vs 8)
- Your model was trained on 8 classes (no "nothing")
- `GESTURES` in `backend/app.py` already has 8 — do not change it

---

## Architecture

```
Browser (index.html)
    │
    ├── GET  /video_feed          → MJPEG stream (30fps)
    ├── GET  /api/state           → JSON state poll (250ms)
    ├── GET  /api/logs            → log entries (600ms)
    ├── POST /api/start_detection → starts detection thread
    ├── POST /api/stop_detection  → stops detection thread
    ├── POST /api/collect         → runs collect_data.py subprocess
    ├── POST /api/train           → runs train_model.py subprocess
    ├── POST /api/speak           → gTTS audio output
    ├── POST /api/clear           → clear sentence
    ├── POST /api/undo            → undo last word
    └── POST /api/gemini          → connect Gemini API

Flask server (app.py)
    ├── Detection thread          → OpenCV + MediaPipe + Transformer
    ├── MJPEG encoder             → annotated frames at 30fps
    ├── Subprocess manager        → collect_data.py / train_model.py
    ├── Async Gemini LLM          → background thread, cached
    └── gTTS speaker              → background thread
```