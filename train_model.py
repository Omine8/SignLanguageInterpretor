import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    Input, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import seaborn as sns


GESTURES        = ["hello", "thanks", "yes", "i", "fine", "please", "sorry", "need"]
DATA_PATH       = "data_v2"
SEQUENCE_LENGTH = 30
INPUT_SIZE      = 150   # 63 hand + 69 pose + 18 face
NUM_CLASSES     = len(GESTURES)


print("=" * 55)
print("  Loading dataset...")
print("=" * 55)

X, y = [], []
label_map = {gesture: i for i, gesture in enumerate(GESTURES)}

for gesture in GESTURES:
    gesture_path = os.path.join(DATA_PATH, gesture)
    if not os.path.exists(gesture_path):
        print(f"  WARNING: folder not found → {gesture_path}")
        continue
    count = 0
    for sample_folder in os.listdir(gesture_path):
        seq_path = os.path.join(gesture_path, sample_folder, "sequence.npy")
        if os.path.exists(seq_path):
            seq = np.load(seq_path)
            if seq.shape == (SEQUENCE_LENGTH, INPUT_SIZE):
                X.append(seq)
                y.append(label_map[gesture])
                count += 1
            else:
                print(f"  WARNING: wrong shape {seq.shape} → {seq_path}")
    print(f"  {gesture:10} : {count} samples loaded")

print(f"\n  Total samples: {len(X)}")
assert len(X) > 0, "No data found! Run collect_data.py first."

X = np.array(X, dtype=np.float32)
y_cat = to_categorical(y, NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y)

print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Injects information about POSITION of each frame into the model.

    Why needed:
    - Transformer has no built-in sense of order (unlike LSTM)
    - Without this, frame 1 and frame 30 look identical to the model
    - We add a unique sine/cosine pattern to each frame position
    - This is the same technique used in the original "Attention Is All You Need" paper

    Shape: (batch, 30 frames, 150 features) → same shape out
    """
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


def transformer_encoder_block(x, num_heads, ff_dim, dropout_rate=0.1):
    
    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=INPUT_SIZE // num_heads,  # 150 / 4 = 37 per head
        dropout=dropout_rate
    )(x, x)   # query=x, key=x, value=x  (self-attention)

    # Residual + LayerNorm
    x = Add()([x, attn_out])
    x = LayerNormalization(epsilon=1e-6)(x)

    # — Feed-Forward —
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dropout(dropout_rate)(ff)
    ff = Dense(INPUT_SIZE)(ff)   # project back to original dimension

    # Residual + LayerNorm
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x


def build_transformer():
    
    inputs = Input(shape=(SEQUENCE_LENGTH, INPUT_SIZE), name="frame_sequence")

    # Add positional info
    x = PositionalEncoding(max_len=SEQUENCE_LENGTH, d_model=INPUT_SIZE)(inputs)
    x = Dropout(0.1)(x)

    # Two stacked Transformer encoder blocks
    x = transformer_encoder_block(x, num_heads=4, ff_dim=256, dropout_rate=0.1)
    x = transformer_encoder_block(x, num_heads=4, ff_dim=256, dropout_rate=0.1)

    # Aggregate across all 30 frames
    x = GlobalAveragePooling1D()(x)

    # Classification head
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax", name="gesture_probs")(x)

    return Model(inputs, outputs, name="SignLanguage_Transformer")


# ─── LEARNING RATE WARMUP ─────────────────────────────────────────────────────
class WarmupCosineSchedule(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=10, total_epochs=100, peak_lr=1e-3):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.peak_lr       = peak_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.peak_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.peak_lr * 0.5 * (1 + np.cos(np.pi * progress))
        tf.keras.backend.set_value(self.model.optimizer.lr, float(lr))


# ─── TRAIN TRANSFORMER ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Building Temporal Transformer...")
print("=" * 55)

transformer = build_transformer()
transformer.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
transformer.summary()

callbacks_transformer = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    WarmupCosineSchedule(warmup_epochs=10, total_epochs=100, peak_lr=1e-3),
]

print("\nTraining Transformer — this takes ~20–30 min on CPU...")
history_t = transformer.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks_transformer,
    verbose=1
)

loss_t, acc_t = transformer.evaluate(X_test, y_test, verbose=0)
print(f"\n  Transformer Test Accuracy: {acc_t * 100:.2f}%")

# ─── SAVE MODEL ───────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
transformer.save("models/sign_transformer_v1.h5")
print("  Saved → models/sign_transformer_v1.h5")

# ─── CLASSIFICATION REPORT ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Per-gesture accuracy breakdown:")
print("=" * 55)
y_pred  = np.argmax(transformer.predict(X_test, verbose=0), axis=1)
y_true  = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=GESTURES))

# ─── SAVE CONFUSION MATRIX PLOT ───────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=GESTURES, yticklabels=GESTURES, ax=ax)
ax.set_title("Confusion Matrix — Temporal Transformer", fontsize=14, pad=15)
ax.set_ylabel("True Gesture")
ax.set_xlabel("Predicted Gesture")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
print("\n  Confusion matrix saved → models/confusion_matrix.png")

# ─── TRAINING CURVE ───────────────────────────────────────────────────────────
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(history_t.history["accuracy"],     label="Train", linewidth=2)
ax1.plot(history_t.history["val_accuracy"], label="Val",   linewidth=2)
ax1.set_title("Accuracy — Temporal Transformer")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history_t.history["loss"],     label="Train", linewidth=2)
ax2.plot(history_t.history["val_loss"], label="Val",   linewidth=2)
ax2.set_title("Loss — Temporal Transformer")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/training_curves.png", dpi=150)
print("  Training curves saved → models/training_curves.png")

print("\n✅  DONE. Files saved:")
print("    models/sign_transformer_v1.h5")
print("    models/confusion_matrix.png")
print("    models/training_curves.png")
print(f"\n    Final test accuracy: {acc_t * 100:.2f}%")