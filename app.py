"""
app.py  —  Flask + SocketIO backend for SignSense
==================================================
Replaces the OpenCV window from sen_form.py.
- Captures webcam frames on the server
- Runs MediaPipe + Transformer model
- Streams predictions to the browser via WebSocket
- Browser sends control commands back (speak / clear / undo)
 
Run:
    python app.py
Then open:
    http://localhost:5000
"""
 
import os
import time
import threading
import numpy as np
from collections import deque
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
 
# ── TensorFlow / Keras ────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import load_model
 
# ── MediaPipe ─────────────────────────────────────────
import mediapipe as mp
import cv2
 
# ══════════════════════════════════════════════════════
#  CONFIG  (mirrors sen_form.py exactly)
# ══════════════════════════════════════════════════════
GESTURES             = ["hello", "thanks", "yes", "no", "i", "fine", "please", "sorry", "need"]
SEQUENCE_LENGTH      = 30
INPUT_SIZE           = 150
CONFIDENCE_THRESHOLD = 0.92
ENTROPY_THRESHOLD    = 1.8
STABLE_FRAMES        = 8
COOLDOWN_SECONDS     = 2.5
MAX_SENTENCE_LEN     = 12
FACE_LANDMARKS       = [1, 152, 33, 263, 61, 291]
POSE_LANDMARKS       = [0,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
 
# ══════════════════════════════════════════════════════
#  POSITIONAL ENCODING  (needed to load the transformer)
# ══════════════════════════════════════════════════════
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len=30, d_model=150, **kwargs):
        super().__init__(**kwargs)
        positions = np.arange(max_len)[:, np.newaxis]
        dims      = np.arange(d_model)[np.newaxis, :]
        angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.cast(angles[np.newaxis, :, :], tf.float32)
    def call(self, x):
        return x + self.pe
 
# ══════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════
print("Loading model...")
model      = None
model_name = "none"
 
for fpath, mname in [("models/sign_transformer_v1.h5", "Transformer"),
                     ("models/sign_lstm_v2.h5", "LSTM")]:
    if not os.path.exists(fpath):
        continue
    try:
        model      = load_model(fpath, compile=False,
                                custom_objects={"PositionalEncoding": PositionalEncoding})
        model_name = mname
        print(f"  Loaded {mname}: {model.output_shape}")
        break
    except Exception as e:
        print(f"  Failed {fpath}: {e}")
 
if model is None:
    print("WARNING: No model found — running in demo mode (random predictions).")
    print("         Run train_model.py first for real predictions.")
 
# ══════════════════════════════════════════════════════
#  MEDIAPIPE SETUP  (Using the stable solutions API)
# ══════════════════════════════════════════════════════
print("Loading MediaPipe models...")
 
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_face  = mp.solutions.face_mesh
 
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
pose  = mp_pose.Pose(min_detection_confidence=0.7,
                     min_tracking_confidence=0.7,
                     model_complexity=0)
face  = mp_face.FaceMesh(max_num_faces=1,
                         min_detection_confidence=0.7,
                         refine_landmarks=False)
 
# ══════════════════════════════════════════════════════
#  KEYPOINT EXTRACTION  (matching sen_form.py)
# ══════════════════════════════════════════════════════
def extract_keypoints(hr, pr, fr):
    hand_kp = np.array([[lm.x, lm.y, lm.z]
                        for lm in hr.multi_hand_landmarks[0].landmark]).flatten() \
              if hr.multi_hand_landmarks else np.zeros(63)
    pose_kp = np.array([[pr.pose_landmarks.landmark[i].x,
                         pr.pose_landmarks.landmark[i].y,
                         pr.pose_landmarks.landmark[i].z]
                        for i in POSE_LANDMARKS]).flatten() \
              if pr.pose_landmarks else np.zeros(len(POSE_LANDMARKS) * 3)
    fl      = fr.multi_face_landmarks[0].landmark if fr.multi_face_landmarks else None
    face_kp = np.array([[fl[i].x, fl[i].y, fl[i].z]
                        for i in FACE_LANDMARKS]).flatten() \
              if fl else np.zeros(len(FACE_LANDMARKS) * 3)
    kp = np.concatenate([hand_kp, pose_kp, face_kp])
    if np.isnan(kp).any() or np.isinf(kp).any():
        kp = np.zeros_like(kp)
    return kp
 
def calc_entropy(probs):
    p = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))
 
# ══════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════
app    = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "signsense-secret"
sio    = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
 
# ══════════════════════════════════════════════════════
#  RECOGNITION STATE  (shared across threads)
# ══════════════════════════════════════════════════════
class RecogState:
    def __init__(self):
        self.sequence       = []
        self.sentence_words = []
        self.stable_buf     = deque(maxlen=STABLE_FRAMES)
        self.last_word      = ""
        self.last_time      = 0.0
        self.lock           = threading.Lock()
 
state = RecogState()
 
# ══════════════════════════════════════════════════════
#  CAMERA  — two-thread architecture
#   Thread 1 (video_loop):  cap.read() → JPEG → emit "frame"  ~20 fps
#   Thread 2 (mediapipe_loop): reads shared frame → MediaPipe + model → emit "prediction"
# ══════════════════════════════════════════════════════
import base64
from threading import Lock
 
_cam_running  = False
_cam_thread   = None
_latest_frame = None          # most-recent BGR frame, shared between threads
_frame_lock   = threading.Lock()
 
# ── Thread 1: capture + stream video ──────────────────
def video_loop(cap):
    global _cam_running, _latest_frame
    frame_count = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 50]
 
    while _cam_running:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame_count += 1
 
        # share latest frame with MediaPipe thread (drop old ones)
        with _frame_lock:
            _latest_frame = frame
 
        # encode + stream at ~20 fps (every other frame)
        if frame_count % 2 == 0:
            small   = cv2.resize(frame, (480, 360))
            display = cv2.flip(small, 1)
            ret_enc, buf = cv2.imencode(".jpg", display, encode_params)
            if ret_enc:
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                sio.emit("frame", {"jpeg": b64})
 
# ── Thread 2: MediaPipe + model prediction ─────────────
def mediapipe_loop():
    global _cam_running, _latest_frame
    last_processed = None      # avoid processing same frame twice
    frame_count    = 0
 
    while _cam_running:
        with _frame_lock:
            frame = _latest_frame
 
        # skip if no new frame
        if frame is None or frame is last_processed:
            time.sleep(0.01)
            continue
        last_processed = frame
        frame_count   += 1
 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hr = hands.process(rgb)
        pr = pose.process(rgb)
        try:
            fr = face.process(rgb)
        except Exception:
            fr = type('obj', (object,), {'multi_face_landmarks': None})()
        rgb.flags.writeable = True
 
        hand_visible = hr.multi_hand_landmarks is not None
 
        with state.lock:
            kp = extract_keypoints(hr, pr, fr)
            state.sequence.append(kp)
            state.sequence = state.sequence[-SEQUENCE_LENGTH:]
 
            cur_word   = ""
            conf       = 0.0
            entropy    = 0.0
            probs_list = [0.0] * len(GESTURES)
            new_word   = None
 
            if not hand_visible:
                state.stable_buf.clear()
 
            elif len(state.sequence) == SEQUENCE_LENGTH:
                try:
                    if model is not None:
                        inp   = np.expand_dims(np.array(state.sequence, dtype=np.float32), 0)
                        preds = model.predict(inp, verbose=0)[0]
                    else:
                        preds = np.random.dirichlet(np.ones(len(GESTURES)))
 
                    conf       = float(np.max(preds))
                    entropy    = calc_entropy(preds)
                    probs_list = preds.tolist()
                    predicted  = GESTURES[int(np.argmax(preds))]
 
                    if frame_count % 30 == 0:
                        print(f"  [{frame_count}] {predicted}  conf={conf:.3f}  H={entropy:.3f}")
 
                    if conf < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
                        state.stable_buf.append("")
                    else:
                        cur_word = predicted
                        state.stable_buf.append(predicted)
 
                    if (len(state.stable_buf) == STABLE_FRAMES
                            and len(set(state.stable_buf)) == 1
                            and state.stable_buf[0] != ""
                            and len(state.sentence_words) < MAX_SENTENCE_LEN):
 
                        confirmed = state.stable_buf[0]
                        now = time.time()
                        if not (confirmed == state.last_word
                                and now - state.last_time < COOLDOWN_SECONDS) \
                           and confirmed not in state.sentence_words:
                            state.sentence_words.append(confirmed)
                            state.last_word = confirmed
                            state.last_time = now
                            state.stable_buf.clear()
                            new_word = confirmed
                            print(f"  ✓ Added: '{confirmed}' → {state.sentence_words}")
 
                except Exception as e:
                    print(f"Prediction error: {e}")
                    import traceback; traceback.print_exc()
                    state.sequence.clear()
 
            payload = {
                "hand_visible":   hand_visible,
                "cur_word":       cur_word,
                "confidence":     round(conf, 4),
                "entropy":        round(entropy, 4),
                "probs":          {g: round(p, 4) for g, p in zip(GESTURES, probs_list)},
                "stable_count":   len([x for x in state.stable_buf if x != ""]),
                "sentence_words": list(state.sentence_words),
                "new_word":       new_word,
                "model_name":     model_name,
            }
 
        sio.emit("prediction", payload)
 
# ── Launcher ───────────────────────────────────────────
def camera_loop():
    global _cam_running
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimize internal buffer lag
 
    if not cap.isOpened():
        sio.emit("error", {"msg": "Cannot open camera"})
        _cam_running = False
        return
 
    for _ in range(5):   # minimal warm-up
        cap.read()
 
    sio.emit("camera_ready", {})
    print("Camera ready")
 
    mp_thread = threading.Thread(target=mediapipe_loop, daemon=True)
    mp_thread.start()
 
    video_loop(cap)      # runs in this thread until _cam_running=False
 
    cap.release()
    print("Camera stopped")
 
# ══════════════════════════════════════════════════════
#  SOCKET EVENTS
# ══════════════════════════════════════════════════════
@sio.on("connect")
def on_connect():
    print("Browser connected")
    emit("init", {
        "gestures":     GESTURES,
        "model_name":   model_name,
        "model_loaded": model is not None,
    })
 
@sio.on("start_camera")
def on_start_camera():
    global _cam_running, _cam_thread
    if _cam_running:
        return
    _cam_running = True
    _cam_thread  = threading.Thread(target=camera_loop, daemon=True)
    _cam_thread.start()
 
@sio.on("stop_camera")
def on_stop_camera():
    global _cam_running
    _cam_running = False
 
@sio.on("undo_word")
def on_undo():
    with state.lock:
        if state.sentence_words:
            removed = state.sentence_words.pop()
            print(f"  Removed: '{removed}' → {state.sentence_words}")
            emit("sentence_update", {"sentence_words": list(state.sentence_words)})
 
@sio.on("clear_all")
def on_clear():
    with state.lock:
        state.sentence_words.clear()
        state.stable_buf.clear()
        state.sequence.clear()
        state.last_word = ""
        state.last_time = 0.0
    emit("sentence_update", {"sentence_words": []})
    print("  Cleared")
 
@sio.on("shutdown")
def on_shutdown():
    global _cam_running
    print("Shutdown requested from browser")
    _cam_running = False
    emit("shutdown_ack", {})
    # give the camera thread a moment to stop, then exit
    def _exit():
        time.sleep(1.5)
        os._exit(0)
    threading.Thread(target=_exit, daemon=True).start()
 
# ══════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")
 
# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  SignSense Flask Server")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    sio.run(app, host="0.0.0.0", port=5000, debug=False)