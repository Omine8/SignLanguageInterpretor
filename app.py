from flask import Flask, Response, jsonify, request
import subprocess
import threading
import cv2

app = Flask(__name__)

# ─── GLOBAL STATE ─────────────────
latest_word = ""
running_mode = None

# ─── RUN DETECTION ────────────────
def run_detection():
    global running_mode
    running_mode = "detect"
    subprocess.run(["python", "sen_form.py"])

# ─── RUN DATA COLLECTION ──────────
def run_collection(gesture):
    global running_mode
    running_mode = "collect"
    subprocess.run(["python", "collect_data.py", gesture])

# ─── RUN TRAINING ────────────────
def run_training():
    global running_mode
    running_mode = "train"
    subprocess.run(["python", "train_model.py"])

# ─── ROUTES ──────────────────────

@app.route("/start_detection")
def start_detection():
    threading.Thread(target=run_detection).start()
    return jsonify({"status": "Detection started"})

@app.route("/collect", methods=["POST"])
def collect():
    gesture = request.json.get("gesture")
    threading.Thread(target=run_collection, args=(gesture,)).start()
    return jsonify({"status": f"Collecting data for {gesture}"})

@app.route("/train")
def train():
    threading.Thread(target=run_training).start()
    return jsonify({"status": "Training started"})

@app.route("/status")
def status():
    return jsonify({"mode": running_mode})

app.run(debug=True)
