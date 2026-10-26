"""Train a Singapore food classifier with transfer learning (EfficientNetV2-B0).

Usage:
    python train.py                 # default: 10 frozen epochs + 10 fine-tune epochs
    python train.py --epochs 5 --fine-tune-epochs 5

Reads images from data/<class_name>/, writes:
    space/model.keras      - trained model (deployed to Hugging Face)
    space/labels.json      - class names in output order
    outputs/history.png    - accuracy/loss curves
    outputs/confusion.png  - confusion matrix on the validation set
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SPACE_DIR = ROOT / "space"
OUT_DIR = ROOT / "outputs"
IMG_SIZE = 224
BATCH = 32
SEED = 42


def load_datasets():
    common = dict(
        validation_split=0.2,
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(DATA_DIR, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(DATA_DIR, subset="validation", **common)
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune), class_names


def compute_class_weights(class_names):
    """Weight each class inversely to its image count so small classes aren't ignored."""
    counts = np.array([len(list((DATA_DIR / c).glob("*.jpg"))) for c in class_names])
    weights = counts.sum() / (len(counts) * counts)
    for c, n, w in zip(class_names, counts, weights):
        print(f"  {c:<16} {n:>4} images  weight {w:.2f}")
    return dict(enumerate(weights))


def build_model(num_classes: int):
    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augment",
    )
    # include_preprocessing=True: the model takes raw 0-255 pixels, so the app needs no extra scaling
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3), include_preprocessing=True
    )
    base.trainable = False

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    x = base(x, training=False)  # keep BatchNorm in inference mode, even when fine-tuning
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def plot_history(histories):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(acc, label="train")
    a1.plot(val_acc, label="validation")
    a1.set_title("Accuracy")
    a2.plot(loss, label="train")
    a2.plot(val_loss, label="validation")
    a2.set_title("Loss")
    for a in (a1, a2):
        a.axvline(len(histories[0].history["loss"]) - 0.5, color="grey", ls="--", lw=1)
        a.set_xlabel("epoch (dashed line = start of fine-tuning)")
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "history.png", dpi=120)


def plot_confusion(model, val_ds, class_names):
    y_true, y_pred = [], []
    for x, y in val_ds:
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(model.predict(x, verbose=0), axis=1))
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(class_names)).numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    labels = [c.replace("_", " ") for c in class_names]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Validation confusion matrix")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion.png", dpi=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--unfreeze", type=int, default=40, help="number of top base layers to fine-tune")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    train_ds, val_ds, class_names = load_datasets()
    print("Classes:")
    class_weight = compute_class_weights(class_names)

    model, base = build_model(len(class_names))
    early_stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    # Phase 1: train only the new classification head
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[early_stop], class_weight=class_weight)

    # Phase 2: unfreeze the top of the base network and fine-tune with a small learning rate
    base.trainable = True
    for layer in base.layers[: -args.unfreeze]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=[early_stop], class_weight=class_weight)

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {acc:.1%}")

    model.save(SPACE_DIR / "model.keras")
    (SPACE_DIR / "labels.json").write_text(json.dumps(class_names, indent=2))
    plot_history([h1, h2])
    plot_confusion(model, val_ds, class_names)
    print(f"Saved model to {SPACE_DIR / 'model.keras'} and charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
