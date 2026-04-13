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
GEMINI_API_KEY = "AIzaSyBtKU4kDd-Xb2gWhDvOkpvut5Bp6oGXsss"
# ══════════════════════════════════════════════════════

# ── Gemini async setup ────────────────────────────────
# Gemini runs in a background thread so it NEVER blocks the camera feed.
# When words change, we instantly show fallback grammar and fire a thread.
# When thread finishes (~1-2s), result appears on screen automatically.
# Same word combo is never sent to Gemini twice (cached).
_llm_available    = False
_gemini_model     = None
_grammar_cache    = {}       # {tuple(words): sentence}  — persists forever
_pending_llm_keys = set()   # keys currently being fetched
_llm_lock         = threading.Lock()

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        _gemini_model.generate_content("Say OK")   # connectivity test
        _llm_available = True
        print("Gemini LLM connected")
    except Exception as e:
        print(f"Gemini failed: {e} — using fallback grammar")

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
# 8 classes — "nothing" removed, retrained
GESTURES             = ["hello", "thanks", "yes", "no", "i", "fine", "please", "sorry"]
SEQUENCE_LENGTH      = 30
INPUT_SIZE           = 150
CONFIDENCE_THRESHOLD = 0.92
ENTROPY_THRESHOLD    = 1.6    # tighter since no idle "nothing" class
STABLE_FRAMES        = 8
COOLDOWN_SECONDS     = 2.5
MAX_SENTENCE_LEN     = 12
FACE_LANDMARKS       = [1, 152, 33, 263, 61, 291]
POSE_LANDMARKS       = [0,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]

# Window layout
WIN_W    = 860
WIN_H    = 480
PANEL_W  = 200    # right confidence panel
CAM_W    = WIN_W - PANEL_W
TOP_H    = 58
BOT_H    = 115

# ── Load model ────────────────────────────────────────
print("Loading model...")
model      = None
model_name = "none"

for fpath, mname in [("models/sign_transformer_v1.h5", "Transformer"),
                     ("models/sign_lstm_v2.h5",        "LSTM")]:
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
    print(f"ERROR: Model outputs {model.output_shape[-1]} classes but GESTURES has {len(GESTURES)}.")
    print(f"Your GESTURES list must exactly match what you trained on.")
    input("Press Enter to exit...")
    exit()

print(f"Model OK — {model_name} — {len(GESTURES)} classes")

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
                     min_tracking_confidence=0.7)
face  = mp_face.FaceMesh(max_num_faces=1,
                          min_detection_confidence=0.7,
                          refine_landmarks=False)
print("MediaPipe OK")

# ── Camera ────────────────────────────────────────────
print("Opening camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIN_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)
if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    input("Press Enter to exit...")
    exit()
print("Camera OK — warming up...")
for _ in range(25):
    cap.read()

# ── Pygame audio ──────────────────────────────────────
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
    if np.isnan(kp).any() or np.isinf(kp).any():
        kp = np.zeros_like(kp)
    return kp

def calc_entropy(probs):
    p = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))

# ── Grammar ───────────────────────────────────────────
def _llm_thread(key):
    try:
        prompt = (
            "Convert these sign language words into one short natural English sentence. "
            "Return ONLY the sentence. No explanation.\n"
            f"Words: {', '.join(key)}"
        )
        result = _gemini_model.generate_content(prompt).text.strip()
        if result:
            with _llm_lock:
                _grammar_cache[key] = result
    except Exception as e:
        print(f"LLM error: {e}")

def make_sentence(words):
    if not words:
        return "..."
    key = tuple(w.lower() for w in words)
    with _llm_lock:
        if key in _grammar_cache:
            return _grammar_cache[key]
    if _llm_available and key not in _pending_llm_keys:
        _pending_llm_keys.add(key)
        threading.Thread(target=_llm_thread, args=(key,), daemon=True).start()
    return _fallback(words)

def _fallback(words):
    fixes = {
        ("i","fine")         : "I am fine.",
        ("i","sorry")        : "I am sorry.",
        ("yes","please")     : "Yes, please.",
        ("no","thanks")      : "No, thank you.",
        ("hello","i","fine") : "Hello! I am fine.",
        ("i","yes")          : "Yes, I do.",
        ("i","no")           : "No, I don't.",
        ("hello","thanks")   : "Hello, thank you.",
        ("i","please")       : "I would like that, please.",
        ("sorry","yes")      : "Yes, sorry about that.",
        ("sorry","no")       : "No, I am sorry.",
        ("yes","i","fine")   : "Yes, I am fine.",
        ("hello","yes")      : "Hello! Yes.",
        ("hello","no")       : "Hello. No.",
        ("yes","thanks")     : "Yes, thank you.",
        ("no","sorry")       : "No, sorry.",
        ("hello","please")   : "Hello, please help me.",
    }
    k = tuple(w.lower() for w in words)
    if k in fixes:
        return fixes[k]
    r = words[0].capitalize()
    if len(words) > 1:
        r += " " + " ".join(words[1:])
    return r.rstrip(".") + "."

# ── Duplicate detection ───────────────────────────────
# Blocks:  same word held too long (cooldown)
#          same word as the very last word just added (consecutive)
# Allows:  same word appearing earlier in sentence ("yes i yes" is valid)
def is_duplicate(confirmed, sentence_words, last_word, last_time):
    if confirmed == last_word and time.time() - last_time < COOLDOWN_SECONDS:
        return True
    if sentence_words and sentence_words[-1] == confirmed:
        return True
    return False

# ── UI drawing ────────────────────────────────────────
def draw_panel(frame, probs):
    px    = WIN_W - PANEL_W + 6
    maxbw = PANEL_W - 14
    bar_h = 20
    top_i = int(np.argmax(probs))
    gap   = max(bar_h + 5, (WIN_H - TOP_H - 16) // len(GESTURES))

    ov = frame.copy()
    cv2.rectangle(ov, (WIN_W - PANEL_W, 0), (WIN_W, WIN_H), (14, 16, 20), -1)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (WIN_W - PANEL_W, 0), (WIN_W - PANEL_W, WIN_H), (42, 48, 58), 1)

    cv2.putText(frame, model_name.upper(), (px, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 130), 1)
    cv2.putText(frame, "CONFIDENCE", (px, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 72, 88), 1)
    cv2.line(frame, (WIN_W - PANEL_W + 4, 42), (WIN_W - 4, 42), (38, 44, 54), 1)

    for i, (g, p) in enumerate(zip(GESTURES, probs)):
        y  = TOP_H + 8 + i * gap
        bw = int(p * maxbw)
        is_top = (i == top_i)
        ok     = is_top and p >= CONFIDENCE_THRESHOLD

        if ok:
            bc, tc, lc = (0, 185, 110), (190, 255, 215), (100, 200, 150)
        elif is_top:
            bc, tc, lc = (0, 110, 210), (140, 185, 255), (90, 140, 200)
        else:
            bc, tc, lc = (38, 46, 58), (85, 96, 112), (60, 72, 88)

        cv2.rectangle(frame, (px, y),       (px + maxbw, y + bar_h), (26, 30, 36), -1)
        cv2.rectangle(frame, (px, y),       (px + maxbw, y + bar_h), (38, 44, 54), 1)
        if bw > 1:
            cv2.rectangle(frame, (px, y),   (px + bw,    y + bar_h), bc, -1)

        cv2.putText(frame, g.upper(), (px + 3, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, lc, 1)
        pct = f"{p*100:.0f}%"
        (pw, _), _ = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
        cv2.putText(frame, pct, (px + maxbw - pw - 2, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, tc, 1)


def draw_top(frame, cur_word, conf, ent, hand_vis):
    cv2.rectangle(frame, (0, 0), (CAM_W - 1, TOP_H), (17, 19, 23), -1)
    cv2.line(frame,      (0, TOP_H), (CAM_W, TOP_H),  (38, 44, 54), 1)

    if not hand_vis:
        cv2.putText(frame, "Show your hand to begin",
                    (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 72, 88), 1)
        cv2.putText(frame, "Waiting for hand detection...",
                    (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (45, 55, 68), 1)
        return

    if cur_word:
        col = (0, 215, 120) if conf >= CONFIDENCE_THRESHOLD else (0, 140, 230)
        cv2.putText(frame, cur_word.upper(), (14, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, col, 2)
        # confidence fill bar
        by = 40
        cv2.rectangle(frame, (14, by), (CAM_W - 16, by + 8), (32, 38, 46), -1)
        fw = int((CAM_W - 30) * min(conf, 1.0))
        fc = (0, 195, 110) if conf >= CONFIDENCE_THRESHOLD else (0, 120, 200)
        cv2.rectangle(frame, (14, by), (14 + fw, by + 8), fc, -1)
        cv2.putText(frame, f"conf {conf:.2f}   entropy {ent:.2f}",
                    (14, by + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (65, 80, 100), 1)
    else:
        if ent > ENTROPY_THRESHOLD:
            cv2.putText(frame, "Unknown gesture",
                        (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (45, 90, 155), 1)
            cv2.putText(frame, f"H={ent:.2f} exceeds threshold — not a trained sign",
                        (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (45, 65, 100), 1)
        else:
            cv2.putText(frame, "Hold gesture steady",
                        (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (55, 75, 105), 1)
            cv2.putText(frame, f"conf {conf:.2f} below {CONFIDENCE_THRESHOLD} threshold",
                        (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (45, 60, 90), 1)


def draw_bottom(frame, words, speaking):
    bt = WIN_H - BOT_H
    cv2.rectangle(frame, (0, bt), (CAM_W - 1, WIN_H), (17, 19, 23), -1)
    cv2.line(frame,      (0, bt), (CAM_W, bt),         (38, 44, 54), 1)

    cv2.putText(frame, "SENTENCE", (12, bt + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (48, 58, 72), 1)

    sentence  = make_sentence(words)
    max_chars = 42

    if len(sentence) > max_chars:
        split = sentence.rfind(" ", 0, max_chars)
        split = split if split != -1 else max_chars
        cv2.putText(frame, sentence[:split], (12, bt + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (225, 230, 242), 2)
        cv2.putText(frame, sentence[split:].strip(), (12, bt + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (225, 230, 242), 2)
    else:
        cv2.putText(frame, sentence, (12, bt + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (225, 230, 242), 2)

    # word pills
    px, py1, py2 = 12, WIN_H - 36, WIN_H - 17
    for word in words:
        (tw, _), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        pw = tw + 14
        if px + pw > CAM_W - 18:
            break
        cv2.rectangle(frame, (px, py1), (px + pw, py2), (35, 52, 74), -1)
        cv2.rectangle(frame, (px, py1), (px + pw, py2), (52, 76, 108), 1)
        cv2.putText(frame, word, (px + 6, py2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 195, 240), 1)
        px += pw + 5

    # hints + status
    hy = WIN_H - 4
    if speaking:
        cv2.putText(frame, "Speaking...", (12, hy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 195, 110), 1)
    else:
        hx = 12
        for k, a in [("SPC","speak"),("C","clear"),("BKSP","undo"),("Q","quit")]:
            cv2.putText(frame, k, (hx, hy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 165, 100), 1)
            (kw,_),_ = cv2.getTextSize(k, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            hx += kw + 1
            cv2.putText(frame, f"={a}  ", (hx, hy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (50, 62, 78), 1)
            (aw,_),_ = cv2.getTextSize(f"={a}  ", cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            hx += aw

    llm = "LLM:ON" if _llm_available else "LLM:OFF"
    lc  = (0, 170, 105) if _llm_available else (70, 82, 98)
    (lw,_),_ = cv2.getTextSize(llm, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
    cv2.putText(frame, llm, (CAM_W - lw - 55, hy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, lc, 1)
    wc = f"{len(words)}/{MAX_SENTENCE_LEN}"
    (ww,_),_ = cv2.getTextSize(wc, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
    cv2.putText(frame, wc, (CAM_W - ww - 6, hy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (55, 66, 82), 1)


def render(frame, words, cur_word, conf, ent, speaking, hand_vis, probs):
    draw_panel(frame, probs)
    draw_top(frame, cur_word, conf, ent, hand_vis)
    draw_bottom(frame, words, speaking)
    return frame

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

print(f"\nReady! ({WIN_W}x{WIN_H})")
print("SPACE=speak | C=clear | BACKSPACE=undo | Q=quit\n")

# ══════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.resize(frame, (WIN_W, WIN_H))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hr = hands.process(rgb)
    pr = pose.process(rgb)
    fr = face.process(rgb)
    rgb.flags.writeable = True

    hand_vis = hr.multi_hand_landmarks is not None

    if hr.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame,
                               hr.multi_hand_landmarks[0],
                               mp_hands.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0,195,115), thickness=2, circle_radius=3),
                               mp_draw.DrawingSpec(color=(0,140,82),  thickness=1, circle_radius=1))
    if pr.pose_landmarks:
        mp_draw.draw_landmarks(frame, pr.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(35,70,52),  thickness=1, circle_radius=1),
                               mp_draw.DrawingSpec(color=(52,110,78), thickness=1, circle_radius=1))

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
            inp        = np.expand_dims(np.array(sequence, dtype=np.float32), 0)
            preds      = model.predict(inp, verbose=0)[0]
            conf_disp  = float(np.max(preds))
            ent_disp   = calc_entropy(preds)
            probs_disp = preds
            predicted  = GESTURES[np.argmax(preds)]

            if conf_disp < CONFIDENCE_THRESHOLD or ent_disp > ENTROPY_THRESHOLD:
                cur_word = ""
                stable_buf.append("")
            else:
                cur_word = predicted
                stable_buf.append(predicted)

            if (len(stable_buf) == STABLE_FRAMES
                    and len(set(stable_buf)) == 1
                    and stable_buf[0] != ""
                    and len(sentence_words) < MAX_SENTENCE_LEN):

                confirmed = stable_buf[0]
                if not is_duplicate(confirmed, sentence_words, last_word, last_time):
                    sentence_words.append(confirmed)
                    last_word = confirmed
                    last_time = time.time()
                    stable_buf.clear()
                    # pre-warm LLM for new combo
                    if _llm_available:
                        k = tuple(w.lower() for w in sentence_words)
                        if k not in _grammar_cache and k not in _pending_llm_keys:
                            _pending_llm_keys.add(k)
                            threading.Thread(target=_llm_thread, args=(k,), daemon=True).start()
                    print(f"  Added: '{confirmed}' → {sentence_words}")

        except Exception as e:
            print(f"Prediction error: {e}")
            sequence.clear()

    render(frame, sentence_words, cur_word, conf_disp, ent_disp,
           _speaking, hand_vis, probs_disp)

    cv2.imshow("SignBridge", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        if sentence_words:
            final = make_sentence(sentence_words)
            print(f"Speaking: '{final}'")
            speak(final)
    elif key == ord("c"):
        sentence_words.clear()
        stable_buf.clear()
        sequence.clear()
        last_word = ""; last_time = 0.0; cur_word = ""
        with _llm_lock:
            _grammar_cache.clear()
        _pending_llm_keys.clear()
        probs_disp = np.zeros(len(GESTURES))
        print("Cleared.")
    elif key == 8:
        if sentence_words:
            removed = sentence_words.pop()
            with _llm_lock:
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