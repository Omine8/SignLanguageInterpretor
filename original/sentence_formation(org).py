import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from gtts import gTTS
import pygame
import tempfile
import os
import threading
import time
from collections import deque

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GESTURES             = ["hello", "thanks", "yes", "no", "please", "i", "fine"]
SEQUENCE_LENGTH      = 30
CONFIDENCE_THRESHOLD = 0.85
COOLDOWN_SECONDS     = 2
STABLE_FRAMES        = 8
MAX_SENTENCE_LEN     = 12

# ─── NLP ──────────────────────────────────────────────────────────────────────
def apply_grammar(words):
    if not words:
        return ""
    result = list(words)
    result[0] = result[0].capitalize()
    deduped = [result[0]]
    for w in result[1:]:
        if w.lower() != deduped[-1].lower():
            deduped.append(w)
    result = deduped
    greetings = {"hello", "hi"}
    if len(result) > 1:
        for i, w in enumerate(result[1:], 1):
            if w.lower() in greetings:
                result.pop(i)
                result.insert(0, w.capitalize())
                result[1] = result[1].lower()
                break
    sentence = " ".join(result)
    last = result[-1].lower()
    if last in {"yes", "no", "fine"}:
        sentence += "."
    elif last in {"hello", "hi", "thanks", "please"}:
        sentence += "."
    else:
        sentence += "."
    return sentence

def contextual_fix(sentence):
    fixes = {
        "please help"    : "Please help me.",
        "please yes"     : "Yes, please.",
        "no thanks"      : "No, thank you.",
        "thanks yes"     : "Yes, thank you.",
        "hello please"   : "Hello, please help me.",
        "hello thanks"   : "Hello, thank you.",
        "yes please"     : "Yes, please.",
        "i fine"         : "I am fine.",
        "hello i fine"   : "Hello, I am fine.",
        "i yes"          : "Yes, I do.",
        "i no"           : "No, I don't.",
        "i please"       : "I would like that, please.",
        "i thanks"       : "I am grateful, thank you.",
    }
    low = sentence.lower().rstrip(".")
    for pattern, replacement in fixes.items():
        if low == pattern:
            return replacement
    return sentence

# ─── SPEECH ───────────────────────────────────────────────────────────────────
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

# ─── MEDIAPIPE ────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw  = mp.solutions.drawing_utils

# ─── MODEL ────────────────────────────────────────────────────────────────────
try:
    model = load_model("models/sign_lstm.h5")
    print(f"Model loaded. Output classes: {model.output_shape[-1]}")
    assert model.output_shape[-1] == len(GESTURES), \
        f"Model has {model.output_shape[-1]} outputs but GESTURES has {len(GESTURES)}. Retrain the model!"
except Exception as e:
    print(f"Model load error: {e}")
    exit()

# ─── KEYPOINTS ────────────────────────────────────────────────────────────────
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        kp = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
        # ✅ Guard against bad values
        if np.isnan(kp).any() or np.isinf(kp).any():
            return np.zeros(63)
        return kp
    return np.zeros(63)

# ─── DISPLAY ──────────────────────────────────────────────────────────────────
def draw_ui(frame, sentence_words, current_word, confidence, speaking):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
    conf_color = (0, 255, 80) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
    label = f"Detecting: {current_word}  ({confidence:.2f})" if current_word else "No hand / low confidence"
    cv2.putText(frame, label, (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)
    cv2.rectangle(frame, (0, h - 100), (w, h), (20, 20, 20), -1)
    sentence_str = apply_grammar(sentence_words) if sentence_words else "..."
    sentence_str = contextual_fix(sentence_str)
    cv2.putText(frame, "Sentence:", (10, h - 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
    cv2.putText(frame, sentence_str, (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    hint = "Speaking..." if speaking else "SPACE=speak  C=clear  BKSP=undo  Q=quit"
    cv2.putText(frame, hint, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
    cv2.putText(frame, f"{len(sentence_words)}/{MAX_SENTENCE_LEN} words",
                (w - 160, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    return frame

# ─── STATE ────────────────────────────────────────────────────────────────────
sequence        = []
sentence_words  = []
stable_buffer   = deque(maxlen=STABLE_FRAMES)
last_added_word = ""
last_added_time = 0
current_word    = ""
confidence_disp = 0.0

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Running — SPACE=speak | C=clear | BACKSPACE=undo | Q=quit")

while cap.isOpened():
    ret, frame = cap.read()

    # ✅ Guard against bad frames
    if not ret or frame is None:
        continue

    # frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    if results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-SEQUENCE_LENGTH:]

    # ✅ Only predict when sequence is full, wrapped in try/except
    if len(sequence) == SEQUENCE_LENGTH:
        try:
            input_seq   = np.expand_dims(np.array(sequence), axis=0)
            prediction  = model.predict(input_seq, verbose=0)[0]
            confidence_disp = float(np.max(prediction))
            predicted   = GESTURES[np.argmax(prediction)]
            current_word = predicted if confidence_disp >= CONFIDENCE_THRESHOLD else ""

            stable_buffer.append(
                predicted if confidence_disp >= CONFIDENCE_THRESHOLD else ""
            )

            if (len(stable_buffer) == STABLE_FRAMES
                    and len(set(stable_buffer)) == 1
                    and stable_buffer[0] != ""
                    and len(sentence_words) < MAX_SENTENCE_LEN):

                confirmed = stable_buffer[0]
                now = time.time()

                if not (confirmed == last_added_word
                        and now - last_added_time < COOLDOWN_SECONDS):
                    sentence_words.append(confirmed)
                    last_added_word = confirmed
                    last_added_time = now
                    stable_buffer.clear()
                    print(f"Added: '{confirmed}' | Buffer: {sentence_words}")

        except Exception as e:
            print(f"Prediction error: {e}")
            sequence.clear()   # reset sequence on error

    frame = draw_ui(frame, sentence_words, current_word,
                    confidence_disp, _speaking)

    cv2.imshow("Sign Language Interpreter", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        if sentence_words:
            final = contextual_fix(apply_grammar(sentence_words))
            print(f"Speaking: {final}")
            speak(final)
    elif key == ord("c"):
        sentence_words.clear()
        stable_buffer.clear()
        last_added_word = ""
        print("Cleared.")
    elif key == 8:
        if sentence_words:
            print(f"Removed: '{sentence_words.pop()}'")

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("Stopped.")