"""
app.py
Streamlit demo: draw a digit and watch a Perceptron, an ANN, and a CNN
each predict what it is.

Run locally:
    streamlit run app.py
"""

import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

st.set_page_config(page_title="1 DIGIT,3 MINDS", page_icon="✏️", layout="centered")


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

col1, col2 = st.columns([1, 1])

with col1:
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

with col2:
    st.subheader("Predictions")
    if canvas_result.image_data is not None and canvas_result.image_data[:, :, :3].sum() > 0:
        arr = preprocess(canvas_result.image_data)
        results = predict_all(models, arr)

        for name, res in results.items():
            st.metric(label=name, value=res["prediction"], delta=f"{res['confidence']*100:.1f}% confident")

        st.divider()
        st.caption("Probability distribution per model")
        for name, res in results.items():
            st.write(f"**{name}**")
            st.bar_chart(res["probs"])
    else:
        st.info("Draw a digit on the left to see predictions.")

st.divider()
st.caption(
    "Built with TensorFlow/Keras + Streamlit. Source & training code on "
    "[GitHub](https://github.com/himanshuydv9214/Number_argue)."
)
