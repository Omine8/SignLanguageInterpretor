import cv2
import numpy as np
import mediapipe as mp
import os

# ─── MEDIAPIPE ────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_face  = mp.solutions.face_mesh

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
pose  = mp_pose.Pose(min_detection_confidence=0.7,
                     min_tracking_confidence=0.7,
                     model_complexity=0)
face  = mp_face.FaceMesh(max_num_faces=1,
                          min_detection_confidence=0.7,
                          refine_landmarks=False)
mp_draw = mp.solutions.drawing_utils

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GESTURES        = ["hello", "thanks", "yes", "no", "i", "fine", "please", "sorry"]
SEQUENCE_LENGTH     = 30
SAMPLES_PER_GESTURE = 30     # only 30 real samples needed
DATA_PATH           = "data_v2"

FACE_LANDMARKS = [1, 152, 33, 263, 61, 291]
POSE_LANDMARKS = [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                  23, 24, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28]

# ─── KEYPOINT EXTRACTION ──────────────────────────────────────────────────────
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
    if np.isnan(kp).any() or np.isinf(kp).any():
        kp = np.zeros_like(kp)
    return kp

# ─── DATA AUGMENTATION ────────────────────────────────────────────────────────
def augment_sequence(seq):
    """
    Takes 1 real sequence, returns 4 augmented versions.
    Total = 1 real + 4 augmented = 5x your data.
    All augmentations are realistic variations a real person would make.
    """
    augmented = []

    # 1. Add tiny random noise (simulates sensor jitter)
    noise = seq + np.random.normal(0, 0.005, seq.shape)
    augmented.append(noise)

    # 2. Slight scale change (simulates hand closer/further from camera)
    scale = seq * np.random.uniform(0.92, 1.08)
    augmented.append(scale)

    # 3. Slight horizontal shift (simulates hand slightly left/right)
    shift = seq.copy()
    shift[:, 0::3] += np.random.uniform(-0.03, 0.03)  # shift x coords
    augmented.append(shift)

    # 4. Time warp — slightly speed up or slow down the gesture
    seq_len = seq.shape[0]
    indices = np.sort(np.random.choice(seq_len, seq_len, replace=True))
    indices = np.clip(np.sort(indices), 0, seq_len - 1)
    warped = seq[indices]
    augmented.append(warped)

    return augmented

# ─── CREATE FOLDERS ───────────────────────────────────────────────────────────
for gesture in GESTURES:
    for sample in range(SAMPLES_PER_GESTURE):
        os.makedirs(os.path.join(DATA_PATH, gesture, str(sample)), exist_ok=True)

# ─── CAMERA SETUP ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

# Warm up
print("Warming up camera...")
for _ in range(20):
    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, "Starting soon — get ready!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Collecting", frame)
        cv2.waitKey(100)

print(f"\nCollecting {SAMPLES_PER_GESTURE} samples × {len(GESTURES)} gestures")
print(f"Each sample auto-generates 4 augmented copies")
print(f"Total training samples will be: {SAMPLES_PER_GESTURE * 5 * len(GESTURES)}\n")

# ─── COLLECTION LOOP ──────────────────────────────────────────────────────────
for gesture in GESTURES:
    print(f"\n{'='*40}")
    print(f"  GESTURE: {gesture.upper()}")
    print(f"{'='*40}")

    # Show gesture name for 3 seconds before starting
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.rectangle(frame, (0, 0), (640, 100), (20, 20, 20), -1)
        cv2.putText(frame, f"NEXT GESTURE: {gesture.upper()}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, "Get ready...", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.imshow("Collecting", frame)
        cv2.waitKey(100)

    for sample in range(SAMPLES_PER_GESTURE):
        sequence = []

        # Countdown
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.rectangle(frame, (0, 0), (640, 100), (20, 20, 20), -1)
            cv2.putText(frame, f"{gesture}  |  Sample {sample+1}/{SAMPLES_PER_GESTURE}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"Starting in {countdown}...",
                        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
            cv2.imshow("Collecting", frame)
            cv2.waitKey(1000)

        # Record 30 frames
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_res = hands.process(rgb)
            pose_res = pose.process(rgb)
            face_res = face.process(rgb)
            rgb.flags.writeable = True

            kp = extract_keypoints(hand_res, pose_res, face_res)
            sequence.append(kp)

            # Draw landmarks
            if hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame,
                    hand_res.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS)
            if pose_res.pose_landmarks:
                mp_draw.draw_landmarks(frame,
                    pose_res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS)

            # Progress bar
            progress = int((frame_num + 1) / SEQUENCE_LENGTH * 200)
            cv2.rectangle(frame, (0, 0), (640, 100), (20, 20, 20), -1)
            cv2.putText(frame, f"RECORDING: {gesture.upper()}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (10, 55), (210, 75), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, 55), (10 + progress, 75), (0, 255, 80), -1)
            cv2.putText(frame, f"Frame {frame_num+1}/{SEQUENCE_LENGTH}",
                        (220, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("Collecting", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # Save real sample
        seq_array = np.array(sequence)
        save_path = os.path.join(DATA_PATH, gesture, str(sample), "sequence.npy")
        np.save(save_path, seq_array)

        # Generate and save augmented samples
        aug_sequences = augment_sequence(seq_array)
        aug_base = SAMPLES_PER_GESTURE + (sample * 4)

        for aug_idx, aug_seq in enumerate(aug_sequences):
            aug_folder = os.path.join(DATA_PATH, gesture, str(aug_base + aug_idx))
            os.makedirs(aug_folder, exist_ok=True)
            np.save(os.path.join(aug_folder, "sequence.npy"), aug_seq)

        print(f"  Saved: {gesture}/sample {sample+1} + 4 augmented copies")

cap.release()
cv2.destroyAllWindows()

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
print("\nData collection complete!")
print("\nSummary:")
for gesture in GESTURES:
    path = os.path.join(DATA_PATH, gesture)
    count = len(os.listdir(path)) if os.path.exists(path) else 0
    print(f"  {gesture:10} : {count} samples")