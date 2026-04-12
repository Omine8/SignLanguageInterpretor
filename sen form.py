import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from gtts import gTTS
import pygame
import tempfile
import os
import threading
import time
from collections import deque

# ══════════════════════════════════════════════════════
GEMINI_API_KEY = ""
# ══════════════════════════════════════════════════════

_llm_available = False
_gemini_model  = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model  = genai.GenerativeModel("gemini-1.5-flash")
        _llm_available = True
        print("Gemini LLM connected")
    except Exception as e:
        print(f"Gemini failed: {e}")

# ── PositionalEncoding ────────────────────────────────
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

# ── Config ────────────────────────────────────────────
GESTURES             = ["hello", "thanks", "yes", "no", "i", "fine", "please", "sorry"]
SEQUENCE_LENGTH      = 30
INPUT_SIZE           = 150
CONFIDENCE_THRESHOLD = 0.80
ENTROPY_THRESHOLD    = 1.8
STABLE_FRAMES        = 8
COOLDOWN_SECONDS     = 2.5
MAX_SENTENCE_LEN     = 12
FACE_LANDMARKS       = [1, 152, 33, 263, 61, 291]
POSE_LANDMARKS       = [0,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]

# ── Load model ────────────────────────────────────────
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
    print("ERROR: No model found. Run train_model.py first.")
    input("Press Enter to exit...")
    exit()

if model.output_shape[-1] != len(GESTURES):
    print(f"ERROR: Model has {model.output_shape[-1]} outputs but need {len(GESTURES)}")
    input("Press Enter to exit...")
    exit()

print(f"Model OK — {model_name}")

# ── MediaPipe ─────────────────────────────────────────
print("Loading MediaPipe...")
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_face  = mp.solutions.face_mesh
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
pose  = mp_pose.Pose(min_detection_confidence=0.7,
                     min_tracking_confidence=0.7,
                     model_complexity=0)
face  = mp_face.FaceMesh(max_num_faces=1,
                          min_detection_confidence=0.7,
                          refine_landmarks=False)
print("MediaPipe OK")

# ── Camera ────────────────────────────────────────────
print("Opening camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    input("Press Enter to exit...")
    exit()
print("Camera OK — warming up...")
for _ in range(25):
    cap.read()

# ── Pygame ────────────────────────────────────────────
pygame.mixer.init()
_speaking = False

def speak(text):
    global _speaking
    if _speaking:
        return
    _speaking = True
    def _run():
        global _speaking
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp.close()
            tts.save(tmp.name)
            pygame.mixer.music.load(tmp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            os.remove(tmp.name)
        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            _speaking = False
    threading.Thread(target=_run, daemon=True).start()

# ── Keypoints ─────────────────────────────────────────
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

    if np.max(kp) != 0:
        kp = kp - np.mean(kp)
        kp = kp / (np.std(kp) + 1e-6)

    return kp

def calc_entropy(probs):
    p = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))

# ── Grammar ───────────────────────────────────────────
# Cache key = tuple so it's hashable and updates when words change
_grammar_cache = {}

def make_sentence(words):
    if not words:
        return "..."
    key = tuple(words)                      # tuple so list changes invalidate cache
    if key in _grammar_cache:
        return _grammar_cache[key]

    if _llm_available:
        try:
            prompt = (
                "Convert these sign language detected words into ONE short natural English sentence. "
                "Do not add extra words. Return ONLY the sentence, no explanation.\n"
                f"Words: {', '.join(words)}"
            )
            result = _gemini_model.generate_content(prompt).text.strip()
            if result:
                _grammar_cache[key] = result
                return result
        except Exception as e:
            print(f"LLM error: {e}")

    # Rule-based fallback
    fixes = {
        ("i", "fine")           : "I am fine.",
        ("i", "sorry")          : "I am sorry.",
        ("yes", "please")       : "Yes, please.",
        ("no", "thanks")        : "No, thank you.",
        ("hello", "i", "fine")  : "Hello! I am fine.",
        ("i", "yes")            : "Yes, I do.",
        ("i", "no")             : "No, I don't.",
        ("hello", "thanks")     : "Hello, thank you.",
        ("please", "help")      : "Please help me.",
        ("i", "please")         : "I would like that, please.",
        ("sorry", "yes")        : "Yes, sorry.",
        ("sorry", "no")         : "No, sorry.",
        ("yes", "i", "fine")    : "Yes, I am fine.",
    }
    k = tuple(w.lower() for w in words)
    if k in fixes:
        result = fixes[k]
        _grammar_cache[key] = result
        return result

    result = words[0].capitalize()
    if len(words) > 1:
        result += " " + " ".join(words[1:])
    result = result.rstrip(".") + "."
    _grammar_cache[key] = result
    return result

# ── Draw confidence bars (right side panel) ───────────
def draw_bars(frame, probs):
    h, w     = frame.shape[:2]
    panel_x  = w - 190
    max_bar  = 170
    bar_h    = 16
    spacing  = max(1, (h - 70) // len(GESTURES))
    top_i    = int(np.argmax(probs))

    # dark background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 5, 5), (w - 2, h - 5), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # header
    cv2.putText(frame, f"Model: {model_name}", (panel_x, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)
    cv2.putText(frame, "Confidence", (panel_x, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)

    for i, (gesture, prob) in enumerate(zip(GESTURES, probs)):
        y  = 45 + i * spacing
        bw = int(prob * max_bar)

        if i == top_i and prob >= CONFIDENCE_THRESHOLD:
            col = (50, 220, 80)    # green — accepted
        elif i == top_i:
            col = (30, 140, 255)   # blue — top but not confident enough
        else:
            col = (65, 85, 105)    # grey — other

        # background track
        cv2.rectangle(frame, (panel_x, y), (panel_x + max_bar, y + bar_h), (38, 38, 38), -1)
        # filled bar
        if bw > 0:
            cv2.rectangle(frame, (panel_x, y), (panel_x + bw, y + bar_h), col, -1)
        # label
        label = f"{gesture[:7]:<7s} {prob:.2f}"
        txt_col = (255, 255, 255) if bw > 28 else (160, 160, 160)
        cv2.putText(frame, label, (panel_x + 2, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, txt_col, 1)

# ── Main UI overlay ───────────────────────────────────
def draw_ui(frame, words, cur_word, conf, ent, speaking, hand_vis, probs):
    h, w = frame.shape[:2]
    panel_x = w - 190

    draw_bars(frame, probs)

    # ── top detection bar ──
    cv2.rectangle(frame, (0, 0), (panel_x - 8, 58), (28, 28, 28), -1)

    if not hand_vis:
        msg = "  Show your hand to begin"
        col = (110, 110, 110)
    elif cur_word:
        col = (50, 220, 80) if conf >= CONFIDENCE_THRESHOLD else (30, 140, 255)
        msg = f"  Detecting: {cur_word.upper()}   conf: {conf:.2f}   entropy: {ent:.2f}"
    else:
        col = (30, 140, 255)
        if ent > ENTROPY_THRESHOLD:
            msg = f"  Uncertain (H={ent:.2f}) — unknown gesture"
        else:
            msg = f"  Confidence too low ({conf:.2f}) — hold gesture steady"

    cv2.putText(frame, msg, (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)

    # ── bottom sentence bar ──
    cv2.rectangle(frame, (0, h - 120), (panel_x - 8, h), (22, 22, 22), -1)

    sentence = make_sentence(words)

    cv2.putText(frame, "Sentence:", (10, h - 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (130, 130, 130), 1)

    # wrap long sentences
    max_chars = 38
    if len(sentence) > max_chars:
        mid = sentence.rfind(" ", 0, max_chars)
        if mid == -1:
            mid = max_chars
        line1 = sentence[:mid]
        line2 = sentence[mid:].strip()
        cv2.putText(frame, line1, (10, h - 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(frame, line2, (10, h - 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    else:
        cv2.putText(frame, sentence, (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2)

    # ── word pills ──
    x = 10
    for word in words:
        sz  = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
        pw  = sz[0] + 14
        if x + pw < panel_x - 15:
            cv2.rectangle(frame, (x, h - 28), (x + pw, h - 10), (50, 72, 95), -1)
            cv2.putText(frame, word, (x + 6, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 215, 255), 1)
            x += pw + 6

    # ── hints ──
    llm_txt = "LLM:ON" if _llm_available else "LLM:OFF"
    llm_col = (50, 220, 80) if _llm_available else (110, 110, 110)
    cv2.putText(frame, llm_txt, (10, h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, llm_col, 1)

    hint = "Speaking..." if speaking else \
           "SPACE=speak  C=clear  BKSP=undo  Q=quit"
    cv2.putText(frame, hint, (75, h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (90, 190, 255), 1)

    cv2.putText(frame, f"{len(words)}/{MAX_SENTENCE_LEN}",
                (panel_x - 58, h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (130, 130, 130), 1)

# ── State ─────────────────────────────────────────────
sequence       = []
sentence_words = []
stable_buf     = deque(maxlen=STABLE_FRAMES)
last_word      = ""
last_time      = 0.0
cur_word       = ""
conf_disp      = 0.0
ent_disp       = 0.0
probs_disp     = np.zeros(len(GESTURES))

print("\nReady!")
print("Controls: SPACE=speak | C=clear | BACKSPACE=undo last word | Q=quit\n")

# ══════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hr = hands.process(rgb)
    pr = pose.process(rgb)
    fr = face.process(rgb)
    rgb.flags.writeable = True

    hand_vis = hr.multi_hand_landmarks is not None

    # landmarks
    if hr.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame,
                               hr.multi_hand_landmarks[0],
                               mp_hands.HAND_CONNECTIONS)
    if pr.pose_landmarks:
        mp_draw.draw_landmarks(frame, pr.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                               mp_draw.DrawingSpec(color=(80,240,121), thickness=1, circle_radius=1))

    # buffer
    kp = extract_keypoints(hr, pr, fr)
    sequence.append(kp)
    sequence = sequence[-SEQUENCE_LENGTH:]

    if not hand_vis:
        stable_buf.clear()
        cur_word   = ""
        conf_disp  = 0.0
        ent_disp   = 0.0
        probs_disp = np.zeros(len(GESTURES))

    elif len(sequence) == SEQUENCE_LENGTH:
        try:
            inp       = np.expand_dims(np.array(sequence, dtype=np.float32), 0)
            preds     = model.predict(inp, verbose=0)[0]
            conf_disp = float(np.max(preds))
            ent_disp  = calc_entropy(preds)
            probs_disp = preds
            predicted = GESTURES[np.argmax(preds)]

            # reject if: nothing class, low confidence, OR high entropy
            if (predicted == "nothing"
                    or conf_disp < CONFIDENCE_THRESHOLD
                    or ent_disp > ENTROPY_THRESHOLD):
                cur_word = ""
                stable_buf.append("")
            else:
                cur_word = predicted
                stable_buf.append(predicted)

            # only add word after STABLE_FRAMES consecutive identical predictions
            if (len(stable_buf) == STABLE_FRAMES
                    and len(set(stable_buf)) == 1
                    and stable_buf[0] != ""
                    and len(sentence_words) < MAX_SENTENCE_LEN):

                confirmed = stable_buf[0]
                now = time.time()

                # cooldown prevents same word repeating too fast
                if not (confirmed == last_word and now - last_time < COOLDOWN_SECONDS):
                    sentence_words.append(confirmed)
                    last_word = confirmed
                    last_time = now
                    stable_buf.clear()        # clear so next word needs fresh 8 frames
                    print(f"  Added: '{confirmed}' → {sentence_words}")

        except Exception as e:
            print(f"Prediction error: {e}")
            sequence.clear()

    draw_ui(frame, sentence_words, cur_word, conf_disp, ent_disp,
            _speaking, hand_vis, probs_disp)

    cv2.imshow("Sign Language Interpreter", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord(" "):
        if sentence_words:
            final = make_sentence(sentence_words)
            print(f"Speaking: '{final}'")
            speak(final)
        else:
            print("Nothing to speak yet.")

    elif key == ord("c"):
        sentence_words.clear()
        stable_buf.clear()
        sequence.clear()
        last_word = ""
        last_time = 0.0
        cur_word  = ""
        _grammar_cache.clear()
        probs_disp = np.zeros(len(GESTURES))
        print("Cleared.")

    elif key == 8:   # BACKSPACE
        if sentence_words:
            removed = sentence_words.pop()
            # invalidate cache for old sentence
            _grammar_cache.clear()
            print(f"Removed: '{removed}' → {sentence_words}")

# ── Cleanup ───────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
hands.close()
pose.close()
face.close()
pygame.mixer.quit()
print("Done.")