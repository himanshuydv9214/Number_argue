"""
train.py
Trains a Perceptron, an ANN, and a CNN on MNIST and saves all three models
to ../saved_models/ so the Streamlit app can load them for inference.

Usage:
    python src/train.py --train_csv data/mnist_train.csv --test_csv data/mnist_test.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, MaxPooling2D

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")


def load_data(train_csv: str, test_csv: str):
    df = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    x = df.drop("label", axis=1).values
    y = df["label"].values

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    x_test = df_test.drop("label", axis=1).values
    y_test = df_test["label"].values

    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 10)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

    return (
        x_train.reshape(-1, 28, 28),
        x_val.reshape(-1, 28, 28),
        x_test.reshape(-1, 28, 28),
        y_train_cat,
        y_val_cat,
        y_test_cat,
        y_test,
    )


def build_perceptron():
    model = keras.Sequential(
        [
            Flatten(input_shape=(28, 28)),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_ann(input_shape):
    model = keras.Sequential(
        [
            layers.Input(input_shape),
            Flatten(),
            layers.Dense(128, activation="relu"),
            Dropout(0.3),
            layers.Dense(64, activation="relu"),
            Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn():
    model = keras.Sequential(
        [
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            layers.Dense(128, activation="relu"),
            Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main(train_csv: str, test_csv: str, epochs: int):
    os.makedirs(MODEL_DIR, exist_ok=True)

    (
        x_train_img,
        x_val_img,
        x_test_img,
        y_train_cat,
        y_val_cat,
        y_test_cat,
        y_test,
    ) = load_data(train_csv, test_csv)

    x_train_cnn = x_train_img.reshape(-1, 28, 28, 1)
    x_val_cnn = x_val_img.reshape(-1, 28, 28, 1)
    x_test_cnn = x_test_img.reshape(-1, 28, 28, 1)

    print("Training Perceptron...")
    perceptron = build_perceptron()
    perceptron.fit(
        x_train_img, y_train_cat,
        validation_data=(x_val_img, y_val_cat),
        epochs=epochs, batch_size=32, verbose=1,
    )

    print("Training ANN...")
    ann = build_ann(x_train_img.shape[1:])
    ann.fit(
        x_train_img, y_train_cat,
        validation_data=(x_val_img, y_val_cat),
        epochs=epochs, batch_size=32, verbose=1,
    )

    print("Training CNN...")
    cnn = build_cnn()
    cnn.fit(
        x_train_cnn, y_train_cat,
        validation_data=(x_val_cnn, y_val_cat),
        epochs=epochs, batch_size=32, verbose=1,
    )

    print("\nEvaluating on test set...")
    for name, model, x in [
        ("Perceptron", perceptron, x_test_img),
        ("ANN", ann, x_test_img),
        ("CNN", cnn, x_test_cnn),
    ]:
        loss, acc = model.evaluate(x, y_test_cat, verbose=0)
        print(f"{name}: test accuracy = {acc:.4f}")

    perceptron.save(os.path.join(MODEL_DIR, "perceptron.keras"))
    ann.save(os.path.join(MODEL_DIR, "ann.keras"))
    cnn.save(os.path.join(MODEL_DIR, "cnn.keras"))
    print(f"\nSaved all three models to {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/mnist_train.csv")
    parser.add_argument("--test_csv", default="data/mnist_test.csv")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    main(args.train_csv, args.test_csv, args.epochs)
