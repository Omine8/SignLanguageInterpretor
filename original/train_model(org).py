import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

GESTURES = ["hello", "thanks", "yes", "no", "please", "i", "fine"]
DATA_PATH = "data"
SEQUENCE_LENGTH = 30

X, y = [], []
label_map = {gesture: i for i, gesture in enumerate(GESTURES)}

for gesture in GESTURES:
    for sample_file in os.listdir(os.path.join(DATA_PATH, gesture)):
        seq = np.load(os.path.join(DATA_PATH, gesture, sample_file))
        X.append(seq)
        y.append(label_map[gesture])

X = np.array(X)                          # shape: (samples, 30, 63)
y = to_categorical(y, len(GESTURES))     # one-hot encoded

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(64,  return_sequences=True, input_shape=(SEQUENCE_LENGTH, 63)),
    Dropout(0.3),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64,  return_sequences=False),
    Dense(64, activation="relu"),
    Dense(len(GESTURES), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {acc*100:.2f}%")

os.makedirs("models", exist_ok=True)
model.save("models/sign_lstm.h5")
print("Model saved.")