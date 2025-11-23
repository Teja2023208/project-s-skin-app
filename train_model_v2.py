import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "models/skin_classifier.h5"

IMG_SIZE = 224
BATCH = 16
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 5

def create_generators():
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        horizontal_flip=True,
        validation_split=0.15
    )

    train_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode="categorical",
        subset="training"
    )

    val_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode="categorical",
        subset="validation"
    )

    return train_gen, val_gen


def build_model(num_classes):
    base = EfficientNetB3(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(base.input, out)
    return model


def main():
    print("Loading dataset...")
    train_gen, val_gen = create_generators()
    num_classes = train_gen.num_classes
    print("Classes detected:", train_gen.class_indices)

    print("Building model...")
    model = build_model(num_classes)

    # Stage 1 – train top layers only
    for layer in model.layers[:-20]:
        layer.trainable = False

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3),
        ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]

    print("\nStage 1 Training...")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_STAGE1, callbacks=callbacks)

    # Stage 2 – unfreeze EfficientNet for fine-tuning
    print("\nUnfreezing model for fine-tuning...")
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nStage 2 Fine-tuning...")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_STAGE2, callbacks=callbacks)

    print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
