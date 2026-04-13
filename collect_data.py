import cv2
import numpy as np
import mediapipe as mp
import os

# ─── MEDIAPIPE ───────────────────────────────────────
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_face  = mp.solutions.face_mesh
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
pose  = mp_pose.Pose(min_detection_confidence=0.7,
                     min_tracking_confidence=0.7)
face  = mp_face.FaceMesh(max_num_faces=1,
                         min_detection_confidence=0.7,
                         refine_landmarks=False)

# ─── CONFIG ──────────────────────────────────────────
# GESTURES = ["hello", "thanks", "yes", "no", "i", "fine", "please", "sorry", "need"]
GESTURES =["need"]

# import sys

# if len(sys.argv) > 1:
#     GESTURES = [sys.argv[1]]
# else:
#     GESTURES = ["hello", "thanks", "yes", "no", "i", "fine", "please", "sorry"]
SEQUENCE_LENGTH     = 30
SAMPLES_PER_GESTURE = 30
DATA_PATH           = "data_v2"

FACE_LANDMARKS = [1, 152, 33, 263, 61, 291]
POSE_LANDMARKS = [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                  23, 24, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28]

# ─── KEYPOINT EXTRACTION ─────────────────────────────
def extract_keypoints(hand_res, pose_res, face_res):
    if hand_res.multi_hand_landmarks:
        hand_kp = np.array([[lm.x, lm.y, lm.z]
                            for lm in hand_res.multi_hand_landmarks[0].landmark]).flatten()
    else:
        hand_kp = np.zeros(63)

    if pose_res.pose_landmarks:
        pose_kp = np.array([[pose_res.pose_landmarks.landmark[i].x,
                             pose_res.pose_landmarks.landmark[i].y,
                             pose_res.pose_landmarks.landmark[i].z]
                            for i in POSE_LANDMARKS]).flatten()
    else:
        pose_kp = np.zeros(len(POSE_LANDMARKS) * 3)

    if face_res.multi_face_landmarks:
        fl = face_res.multi_face_landmarks[0].landmark
        face_kp = np.array([[fl[i].x, fl[i].y, fl[i].z]
                            for i in FACE_LANDMARKS]).flatten()
    else:
        face_kp = np.zeros(len(FACE_LANDMARKS) * 3)

    kp = np.concatenate([hand_kp, pose_kp, face_kp])

    # 🔥 NORMALIZATION (IMPORTANT FOR ACCURACY)
    if np.max(kp) != 0:
        kp = kp - np.mean(kp)
        kp = kp / (np.std(kp) + 1e-6)

    return kp

# ─── AUGMENTATION ────────────────────────────────────
def augment_sequence(seq):
    augmented = []

    noise = seq + np.random.normal(0, 0.005, seq.shape)
    augmented.append(noise)

    scale = seq * np.random.uniform(0.92, 1.08)
    augmented.append(scale)

    shift = seq.copy()
    shift[:, 0::3] += np.random.uniform(-0.03, 0.03)
    augmented.append(shift)

    seq_len = seq.shape[0]
    indices = np.sort(np.random.choice(seq_len, seq_len, replace=True))
    warped = seq[indices]
    augmented.append(warped)

    return augmented

# ─── CREATE BASE FOLDERS ─────────────────────────────
for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_PATH, gesture), exist_ok=True)

# ─── CAMERA ──────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Warming up camera...")
for _ in range(20):
    cap.read()

# ─── COLLECTION LOOP ─────────────────────────────────
for gesture in GESTURES:
    print(f"\n===== {gesture.upper()} =====")

    gesture_path = os.path.join(DATA_PATH, gesture)

    # 🔥 FIND EXISTING DATA
    existing = [int(d) for d in os.listdir(gesture_path) if d.isdigit()]
    start_index = max(existing) + 1 if existing else 0

    print(f"Starting from index: {start_index}")

    for sample in range(start_index, start_index + SAMPLES_PER_GESTURE):
        sequence = []

        # Countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.putText(frame, f"{gesture} | Starting in {i}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)
            cv2.imshow("Collecting", frame)
            cv2.waitKey(1000)

        # Record sequence
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_res = hands.process(rgb)
            pose_res = pose.process(rgb)
            face_res = face.process(rgb)

            kp = extract_keypoints(hand_res, pose_res, face_res)
            sequence.append(kp)

            if hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame,
                    hand_res.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"{gesture} | {frame_num+1}/{SEQUENCE_LENGTH}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            cv2.imshow("Collecting", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        seq_array = np.array(sequence)

        # SAVE REAL SAMPLE
        sample_folder = os.path.join(gesture_path, str(sample))
        os.makedirs(sample_folder, exist_ok=True)
        np.save(os.path.join(sample_folder, "sequence.npy"), seq_array)

        # 🔥 SAVE AUGMENTED DATA (UNIQUE INDEXING)
        aug_sequences = augment_sequence(seq_array)
        aug_index = sample + 1

        for aug_seq in aug_sequences:
            aug_folder = os.path.join(gesture_path, str(aug_index))
            os.makedirs(aug_folder, exist_ok=True)
            np.save(os.path.join(aug_folder, "sequence.npy"), aug_seq)
            aug_index += 1

        print(f"Saved sample {sample} (+ augmented)")

cap.release()
cv2.destroyAllWindows()

# ─── SUMMARY ─────────────────────────────────────────
print("\nData collection complete!\n")

for gesture in GESTURES:
    path = os.path.join(DATA_PATH, gesture)
    count = len(os.listdir(path))
    print(f"{gesture:10} : {count} samples")
