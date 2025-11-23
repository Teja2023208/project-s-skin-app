import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# ----------------------------
# Paths
# ----------------------------
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "models/skin_classifier.h5"

# ----------------------------
# Fast training settings
# ----------------------------
IMG_SIZE = 160          # Fast but still accurate
BATCH_SIZE = 8          # Low RAM usage
EPOCHS_STAGE1 = 5       # Fast freeze training
EPOCHS_STAGE2 = 5       # Light fine-tuning
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4

# ----------------------------
# Data Augmentation (FAST)
# ----------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    validation_split=0.15
)

val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15
)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = val_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

num_classes = train_data.num_classes

print("\nDetected classes →", train_data.class_indices)

# ----------------------------
# Build Model
# ----------------------------
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# FIRST: freeze EfficientNet for fast training
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ----------------------------
# Stage 1: Train Final Layers (FAST)
# ----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_STAGE1),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🚀 Stage 1: Training (FREEZE EfficientNet)...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE1
)

# ----------------------------
# Stage 2: Light Fine-Tuning (HIGH ACCURACY)
# ----------------------------
# Unfreeze top layers only → fast & accurate
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_STAGE2),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🔥 Stage 2: Fine-tuning top 20 layers...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE2
)

# ----------------------------
# Save Model
# ----------------------------
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)

print(f"\n🎉 Training complete! Model saved to: {MODEL_PATH}")
