import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import smart_resize, ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical


def parse_args():
    p = argparse.ArgumentParser(description="Galamo TF training script for fluxaa/orc")
    p.add_argument("--h5-file", type=str, default="galaxy.h5", help="Path to galaxy.h5 file")
    p.add_argument("--checkpoint-dir", type=str, required=True, help="Directory for checkpoints")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    p.add_argument("--epochs", type=int, default=25, help="Total number of epochs to train")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    return p.parse_args()

def load_data(h5_file_path: str, max_samples: int = 2000):
    print(f"[galamo_tf] Loading data from {h5_file_path}")
    with h5py.File(h5_file_path, "r") as f:
        print("[galamo_tf] Keys in file:", list(f.keys()))
        images = f["images"][:max_samples]
        ans = f["ans"][:max_samples]

    print(f"[galamo_tf] Using {images.shape[0]} samples")

    # Resize and normalize
    resized_images = np.array([smart_resize(img, (128, 128)) for img in images])
    images_normalized = resized_images.astype("float32") / 255.0

    num_classes = len(np.unique(ans))
    labels_encoded = to_categorical(ans, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        images_normalized, labels_encoded, test_size=0.2, random_state=42
    )

    print(f"[galamo_tf] X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, num_classes

def build_model(num_classes: int):
    print(f"[galamo_tf] Building model with {num_classes} classes")
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def get_generators(X_train, y_train, batch_size: int):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    return train_generator


def save_checkpoint(checkpoint_dir: Path, epoch: int, model):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "model.keras"
    state_path = checkpoint_dir / "training_state.json"

    print(f"[galamo_tf] Saving model -> {model_path}")
    model.save(model_path)

    state = {"epoch": epoch}
    with state_path.open("w") as f:
        json.dump(state, f)
    print(f"[galamo_tf] Saved training state -> {state_path}")


def load_checkpoint(checkpoint_dir: Path):
    model_path = checkpoint_dir / "model.keras"
    state_path = checkpoint_dir / "training_state.json"

    if not model_path.exists() or not state_path.exists():
        print("[galamo_tf] No checkpoint found, starting from scratch.")
        return None, 0

    print(f"[galamo_tf] Loading model from {model_path}")
    model = load_model(model_path)

    with state_path.open("r") as f:
        state = json.load(f)
    start_epoch = int(state.get("epoch", 0))

    print(f"[galamo_tf] Resuming from epoch {start_epoch}")
    return model, start_epoch


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)

    X_train, X_test, y_train, y_test, num_classes = load_data(args.h5_file, max_samples=2000)


    # Save "encoder" equivalent (num_classes) just like your notebook
    encoder_path = checkpoint_dir / "encoder.pkl"
    with encoder_path.open("wb") as f:
        pickle.dump(num_classes, f)
    print(f"[galamo_tf] Saved encoder info -> {encoder_path}")

    # Either load model from checkpoint or build a new one
    if args.resume:
        model, start_epoch = load_checkpoint(checkpoint_dir)
        if model is None:
            model = build_model(num_classes)
            start_epoch = 0
    else:
        model = build_model(num_classes)
        start_epoch = 0

    train_generator = get_generators(X_train, y_train, args.batch_size)

    total_epochs = args.epochs
    print(f"[galamo_tf] Training from epoch {start_epoch+1} to {total_epochs}")

    for epoch in range(start_epoch, total_epochs):
        print(f"[galamo_tf] Epoch {epoch + 1}/{total_epochs}")

        history = model.fit(
            train_generator,
            epochs=1,
            validation_data=(X_test, y_test),
            verbose=1,
        )

        train_acc = history.history.get("accuracy", [None])[-1]
        val_acc = history.history.get("val_accuracy", [None])[-1]
        print(f"[galamo_tf] Epoch {epoch+1} done. Train acc={train_acc}, Val acc={val_acc}")

        save_checkpoint(checkpoint_dir, epoch + 1, model)

    # Final test evaluation
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[galamo_tf] Final test accuracy: {acc * 100:.2f}%")

    # Save final model as well
    final_model_path = checkpoint_dir / "final_model.keras"
    model.save(final_model_path)
    print(f"[galamo_tf] Saved final model -> {final_model_path}")

    print("[galamo_tf] Training completed successfully.")


if __name__ == "__main__":
    main()
