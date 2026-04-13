from flask import Flask, Response, jsonify, request, render_template
import subprocess
import threading
import cv2
import sys

app = Flask(__name__)

mode = "idle"

# Camera (only for preview stream)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        cv2.putText(frame, mode, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def run_script(script, args=None):
    global mode
    try:
        cmd = [sys.executable, script]
        if args:
            cmd += args
        subprocess.run(cmd)
    except Exception as e:
        print("ERROR:", e)
    mode = "idle"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start_detection")
def start_detection():
    global mode
    if mode != "idle":
        return jsonify({"status": "Already running"})

    mode = "Running Detection"
    threading.Thread(target=run_script, args=("sen_form.py",)).start()

    return jsonify({"status": mode})

@app.route("/collect", methods=["POST"])
def collect():
    global mode
    gesture = request.json.get("gesture")

    if not gesture:
        return jsonify({"status": "Enter gesture name"})

    if mode != "idle":
        return jsonify({"status": "Already running"})

    mode = f"Collecting {gesture}"
    threading.Thread(target=run_script, args=("collect_data.py", [gesture])).start()

    return jsonify({"status": mode})

@app.route("/train")
def train():
    global mode
    if mode != "idle":
        return jsonify({"status": "Already running"})

    mode = "Training Model"
    threading.Thread(target=run_script, args=("train_model.py",)).start()

    return jsonify({"status": mode})

@app.route("/status")
def status():
    return jsonify({"mode": mode})

if __name__ == "__main__":
    app.run(debug=True)