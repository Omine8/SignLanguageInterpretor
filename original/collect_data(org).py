import cv2
import numpy as np
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# GESTURES = ["hello", "thanks", "yes", "no", "please", "i", "fine"]
GESTURES = ["how", "you"]
SEQUENCE_LENGTH = 30
SAMPLES_PER_GESTURE = 50
DATA_PATH = "data"

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
    return np.zeros(63)

for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_PATH, gesture), exist_ok=True)

cap = cv2.VideoCapture(0)

for gesture in GESTURES:
    for sample in range(SAMPLES_PER_GESTURE):
        sequence = []
        print(f"Collecting: {gesture} | Sample {sample+1}/{SAMPLES_PER_GESTURE}")
        cv2.waitKey(2000)  # 2s pause between samples

        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            cv2.putText(frame, f"{gesture} | Sample {sample+1} | Frame {frame_num}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None, mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Collecting", frame)
            cv2.waitKey(1)

        np.save(os.path.join(DATA_PATH, gesture, f"{sample}.npy"), sequence)

cap.release()
cv2.destroyAllWindows()