"""
app.py
Streamlit demo: draw a digit and watch a Perceptron, an ANN, and a CNN
each predict what it is.

Run locally:
    streamlit run app.py
"""

import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

st.set_page_config(page_title="MNIST: 3 Models, 1 Digit", page_icon="✏️", layout="centered")


@st.cache_resource
def load_models():
    perceptron = tf.keras.models.load_model(os.path.join(MODEL_DIR, "perceptron.keras"))
    ann = tf.keras.models.load_model(os.path.join(MODEL_DIR, "ann.keras"))
    cnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "cnn.keras"))
    return perceptron, ann, cnn


def preprocess(image_data: np.ndarray) -> np.ndarray:
    """Canvas RGBA array -> normalized 28x28 grayscale array matching MNIST format."""
    img = Image.fromarray(image_data.astype("uint8"), mode="RGBA").convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    return arr


def preprocess_camera_frame(image: Image.Image) -> np.ndarray | None:
    """
    Take a photo from st.camera_input, use OpenCV to find the digit,
    crop/center it, and return a 28x28 array matching MNIST format.
    Returns None if no digit-like contour is found.
    """
    frame = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # MNIST digits are white-on-black; most photos are dark-ink-on-paper,
    # so we invert + threshold to match that convention.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Assume the largest contour is the digit
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < 10 or h < 10:  # too small to be a real digit
        return None

    digit = thresh[y:y + h, x:x + w]

    # Pad to a square so resizing to 28x28 doesn't distort the digit
    size = max(w, h) + 20
    square = np.zeros((size, size), dtype=np.uint8)
    x_off = (size - w) // 2
    y_off = (size - h) // 2
    square[y_off:y_off + h, x_off:x_off + w] = digit

    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    return resized.astype("float32") / 255.0


def predict_all(models, arr: np.ndarray):
    perceptron, ann, cnn = models
    flat_input = arr.reshape(1, 28, 28)
    cnn_input = arr.reshape(1, 28, 28, 1)

    results = {}
    for name, model, x in [
        ("Perceptron", perceptron, flat_input),
        ("ANN", ann, flat_input),
        ("CNN", cnn, cnn_input),
    ]:
        probs = model.predict(x, verbose=0)[0]
        results[name] = {
            "prediction": int(np.argmax(probs)),
            "confidence": float(np.max(probs)),
            "probs": probs,
        }
    return results


st.title("✏️ MNIST: 3 Models, 1 Digit")
st.write(
    "Draw a digit (0-9) below and see how a **Perceptron**, a **fully-connected ANN**, "
    "and a **CNN** each interpret it — including where the simpler models get fooled."
)

if not os.path.exists(os.path.join(MODEL_DIR, "cnn.keras")):
    st.error(
        "No trained models found in `saved_models/`. Run `python src/train.py` "
        "locally first, then commit the `.keras` files (or point this app at them)."
    )
    st.stop()

models = load_models()

input_mode = st.radio("Input method", ["Draw", "Camera"], horizontal=True)

col1, col2 = st.columns([1, 1])
arr = None

with col1:
    if input_mode == "Draw":
        st.subheader("Draw here")
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas_result.image_data is not None and canvas_result.image_data[:, :, :3].sum() > 0:
            arr = preprocess(canvas_result.image_data)

    else:
        st.subheader("Show a digit to your camera")
        st.caption("Write a large digit on paper and hold it up — dark ink on light paper works best.")
        camera_photo = st.camera_input("Capture", label_visibility="collapsed")
        if camera_photo is not None:
            image = Image.open(camera_photo)
            arr = preprocess_camera_frame(image)
            if arr is None:
                st.warning("Couldn't find a clear digit in the frame. Try better lighting or a bigger digit.")

with col2:
    st.subheader("Predictions")
    if arr is not None:
        st.image(arr, caption="What the model sees (28x28)", width=140)
        results = predict_all(models, arr)

        for name, res in results.items():
            st.metric(label=name, value=res["prediction"], delta=f"{res['confidence']*100:.1f}% confident")

        st.divider()
        st.caption("Probability distribution per model")
        for name, res in results.items():
            st.write(f"**{name}**")
            st.bar_chart(res["probs"])
    else:
        st.info("Draw or capture a digit on the left to see predictions.")

st.divider()
st.caption(
    "Built with TensorFlow/Keras + Streamlit. Source & training code on "
    "[GitHub](https://github.com/YOUR_USERNAME/mnist-digit-classifier)."
)
