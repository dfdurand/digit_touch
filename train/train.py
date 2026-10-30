#!/usr/bin/env python3
"""
digit_touch: training the CNN on MNIST (in float32, on CPU).

Usage:
    source ~/venvs/tf/bin/activate
    python train.py

Produces in train/out/:
    model.keras        the trained model (float32),
    curves.png         loss and accuracy over the epochs
    confusion.png      confusion matrix on the 10,000 test images
    errors.png         examples of misrecognized test images
    summary.json       key figures (accuracy, number of parameters, duration)
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # fewer technical messages from TF
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")    # project decision: CPU only

import matplotlib
matplotlib.use("Agg")                                # no window: we write PNGs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

OUT = Path(__file__).resolve().parent / "out"
SEED = 42            # fixed random seed: two runs give the same result
EPOCHS = 15
BATCH = 128
VAL_FRACTION = 0.1   # 10 % of the training images set aside for validation


# --------------------------------------------------------------------------
# 1. The data
# --------------------------------------------------------------------------
def load_data():
    # Downloaded only once (~11 MB) to ~/.keras/datasets/mnist.npz
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Pixels 0..255 (uint8) -> 0.0..1.0 (float32). Networks learn better with
    # small values of a homogeneous order of magnitude.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # (N, 28, 28) -> (N, 28, 28, 1): a convolution expects a "channels" axis
    # (only 1 here: grayscale; it would be 3 for an RGB image).
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # Training / validation split. The test set (10,000 images) is used only
    # once, at the very end.
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_train))
    n_val = int(len(x_train) * VAL_FRACTION)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (x_train[train_idx], y_train[train_idx],
            x_train[val_idx], y_train[val_idx],
            x_test, y_test)


# --------------------------------------------------------------------------
# 2. Data augmentation
# --------------------------------------------------------------------------
# At each epoch, each training image is slightly distorted at random. The
# model never sees exactly the same image twice, which forces it to learn the
# *shape* of the digit rather than its exact pixels. This is what will
# prepare it for finger drawings, never perfectly straight or centered.
#
# fill_mode="constant" + fill_value=0: the areas that appear at the edge after
# a rotation or shift are filled with BLACK, like the MNIST background.
# (The default "reflect" mode would copy mirrored pieces of the digit.)
augment = keras.Sequential([
    layers.RandomRotation(10 / 360, fill_mode="constant", fill_value=0.0),   # ±10°
    layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0.0),  # ±10 % (±2-3 px)
    layers.RandomZoom(0.1, fill_mode="constant", fill_value=0.0),            # ±10 %
], name="augmentation")


def make_datasets(x_tr, y_tr, x_val, y_val):
    train_ds = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
                .shuffle(len(x_tr), seed=SEED)           # different order at each epoch
                .batch(BATCH)
                .map(lambda x, y: (augment(x, training=True), y),
                     num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    # No augmentation for validation: we want to measure the model on
    # "normal" images.
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH)
    return train_ds, val_ds


# --------------------------------------------------------------------------
# 3. The model (architecture justified in §3 of SUIVI_PROJET.md)
# --------------------------------------------------------------------------
# Augmentation is deliberately NOT a layer of the model: it stays in the data
# pipeline, and the exported .tflite will only contain the layers needed for
# inference.
def build_model():
    return keras.Sequential([
        keras.Input(shape=(28, 28, 1), name="image"),
        layers.Conv2D(8, 3, padding="same", activation="relu", name="conv1"),   # -> 28x28x8
        layers.MaxPooling2D(2, name="pool1"),                                   # -> 14x14x8
        layers.Conv2D(16, 3, padding="same", activation="relu", name="conv2"),  # -> 14x14x16
        layers.MaxPooling2D(2, name="pool2"),                                   # -> 7x7x16
        layers.Flatten(name="flatten"),                                         # -> 784
        layers.Dropout(0.25, name="dropout"),        # active only during training
        layers.Dense(32, activation="relu", name="dense1"),                     # -> 32
        layers.Dense(10, activation="softmax", name="probas"),                  # -> 10 probabilities
    ], name="digit_cnn")


# --------------------------------------------------------------------------
# 4. Diagnostic plots
# --------------------------------------------------------------------------
def plot_curves(hist, path):
    ep = range(1, len(hist["loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, hist["loss"], "o-", label="training (distorted images)")
    a1.plot(ep, hist["val_loss"], "o-", label="validation")
    a1.set_title("Loss (error) — should decrease"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(ep, np.array(hist["accuracy"]) * 100, "o-", label="training (distorted images)")
    a2.plot(ep, np.array(hist["val_accuracy"]) * 100, "o-", label="validation")
    a2.set_title("Accuracy (%) — should increase"); a2.set_xlabel("epoch"); a2.legend()
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    # Log scale to see the few errors next to the ~1000 correct answers
    im = ax.imshow(cm + 1, cmap="Blues", norm=matplotlib.colors.LogNorm())
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if i == j else "black")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("digit PREDICTED by the model"); ax.set_ylabel("TRUE digit")
    ax.set_title("Confusion matrix (MNIST test)\ndiagonal = correct answers")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_errors(x, y_true, probs, path, n=24):
    y_pred = probs.argmax(1)
    wrong = np.where(y_pred != y_true)[0]
    # The most "confident" errors first: the most instructive ones
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:n]
    cols = 8
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.7))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        ax.imshow(x[i, :, :, 0], cmap="gray")
        ax.set_title(f"true {y_true[i]} / read {y_pred[i]}\n{probs[i].max() * 100:.0f} %", fontsize=7)
    fig.suptitle("Model errors on the test set (most confident first)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------
# 5. The main program
# --------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    keras.utils.set_random_seed(SEED)

    x_tr, y_tr, x_val, y_val, x_te, y_te = load_data()
    print(f"Images: {len(x_tr)} training, {len(x_val)} validation, {len(x_te)} test")
    train_ds, val_ds = make_datasets(x_tr, y_tr, x_val, y_val)

    model = build_model()
    model.summary()

    # - optimizer "adam": the method that corrects the weights after each batch
    #   (an improved version of gradient descent that adapts its own step
    #   size).
    # - loss "sparse_categorical_crossentropy": the standard error for
    #   classification. It is low when the probability given to the RIGHT
    #   digit is close to 1. "sparse" = the labels are integers 0..9.
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        # If the validation loss stops improving for 2 epochs, halve the
        # learning rate to fine-tune the weights.
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        # If it stops improving for 4 epochs, stop and keep the best weights
        # (protection against overfitting).
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True,
                                      verbose=1),
    ]

    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks,
                     verbose=2)
    duration = time.time() - t0

    # Final evaluation, only once, on never-seen images
    test_loss, test_acc = model.evaluate(x_te, y_te, verbose=0)
    probs = model.predict(x_te, verbose=0)
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y_te, y_pred)

    model.save(OUT / "model.keras")
    plot_curves(hist.history, OUT / "curves.png")
    plot_confusion(cm, OUT / "confusion.png")
    plot_errors(x_te, y_te, probs, OUT / "errors.png")

    # The most frequent confusions (off-diagonal)
    off = [(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j and cm[i, j]]
    off.sort(reverse=True)
    per_class = {str(i): round(100 * cm[i, i] / cm[i].sum(), 2) for i in range(10)}

    summary = {
        "parameters": int(model.count_params()),
        "epochs_run": len(hist.history["loss"]),
        "training_duration_s": round(duration, 1),
        "test_accuracy_pct": round(100 * test_acc, 2),
        "test_loss": round(float(test_loss), 4),
        "test_errors": int((y_pred != y_te).sum()),
        "accuracy_per_digit_pct": per_class,
        "top_confusions": [f"{i} read as {j}: {n} times" for n, i, j in off[:6]],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n================ SUMMARY ================")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Files written to {OUT}")


if __name__ == "__main__":
    main()
