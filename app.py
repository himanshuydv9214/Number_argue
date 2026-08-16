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

st.set_page_config(
    page_title="1 Digit, 3 Minds",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Theme / model metadata
# --------------------------------------------------------------------------- #

MODEL_META = {
    "Perceptron": {
        "color": "#22d3ee",
        "glow": "34, 211, 238",
        "icon": "◉",
        "tagline": "One layer. No hidden units. Pure linear separation.",
        "accuracy": "~92%",
        "arch": [
            "Flatten(28 × 28) → 784",
            "Dense(10, softmax)",
        ],
        "note": (
            "The simplest possible classifier — it draws 10 straight decision "
            "boundaries through 784-dimensional space. It has no way to "
            "represent curves, loops or strokes, so it leans on raw pixel "
            "overlap and gets confused by anything drawn off-center or at an "
            "unusual slant."
        ),
    },
    "ANN": {
        "color": "#a855f7",
        "glow": "168, 85, 247",
        "icon": "◈",
        "tagline": "Two hidden layers learn features — but ignore geometry.",
        "accuracy": "~97%",
        "arch": [
            "Flatten(28 × 28) → 784",
            "Dense(128, relu) + Dropout(0.3)",
            "Dense(64, relu) + Dropout(0.3)",
            "Dense(10, softmax)",
        ],
        "note": (
            "Hidden layers let it compose pixel patterns into reusable "
            "features, which is a big jump over the Perceptron. But flattening "
            "throws away the fact that neighbouring pixels are related — shift "
            "your digit a few pixels and it has to relearn it from scratch."
        ),
    },
    "CNN": {
        "color": "#34d399",
        "glow": "52, 211, 153",
        "icon": "◆",
        "tagline": "Convolutions see strokes, edges and shape — not pixels.",
        "accuracy": "~99%",
        "arch": [
            "Conv2D(32, 3×3, relu) → MaxPool(2×2)",
            "Conv2D(64, 3×3, relu) → MaxPool(2×2)",
            "Flatten → Dense(128, relu) + Dropout(0.5)",
            "Dense(10, softmax)",
        ],
        "note": (
            "Filters slide across the image looking for edges and curves "
            "wherever they appear, so a digit shifted or slightly rotated "
            "still triggers the same features. That translation tolerance is "
            "why it stays confident on messy hand-drawn input."
        ),
    },
}


def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp {
    background:
        radial-gradient(900px 500px at 12% -8%, rgba(34,211,238,0.13), transparent 60%),
        radial-gradient(800px 500px at 88% 0%, rgba(168,85,247,0.13), transparent 60%),
        radial-gradient(900px 600px at 50% 110%, rgba(52,211,153,0.10), transparent 60%),
        #05060f;
    color: #e8ecf8;
}
html, body, [class*="css"], .stApp, p, span, div, label {
    font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}

section[data-testid="stSidebar"] > div {
    background: rgba(10, 12, 26, 0.86);
    border-right: 1px solid rgba(255,255,255,0.07);
    backdrop-filter: blur(14px);
}

/* ---------------- hero ---------------- */
.hero {
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 22px;
    padding: 30px 34px;
    margin-bottom: 22px;
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015));
    backdrop-filter: blur(18px);
    box-shadow: 0 24px 60px rgba(0,0,0,0.45);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content:""; position:absolute; inset:0;
    background: linear-gradient(90deg, #22d3ee, #a855f7, #34d399);
    height:2px; top:0; opacity:.85;
}
.hero-badge {
    display:inline-block; font-family:'JetBrains Mono', monospace; font-size:.68rem;
    letter-spacing:.22em; text-transform:uppercase; color:#8b93ad;
    border:1px solid rgba(255,255,255,0.14); border-radius:999px;
    padding:5px 13px; margin-bottom:14px;
}
.hero h1 {
    font-size: 2.9rem; font-weight: 700; line-height:1.05; margin:0 0 12px 0;
    background: linear-gradient(100deg, #ffffff 8%, #22d3ee 42%, #a855f7 72%, #34d399 96%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {color:#9aa3bd; font-size:1.02rem; margin:0; max-width:62ch; line-height:1.6;}

/* ---------------- generic panel ---------------- */
.panel {
    border:1px solid rgba(255,255,255,0.08); border-radius:18px;
    background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
    backdrop-filter: blur(14px); padding:18px 20px; margin-bottom:16px;
}
.section-label {
    font-family:'JetBrains Mono', monospace; font-size:.7rem; letter-spacing:.2em;
    text-transform:uppercase; color:#7d86a1; margin:0 0 12px 0;
}

/* ---------------- prediction cards ---------------- */
.pcard {
    position:relative; border-radius:18px; padding:16px 18px; margin-bottom:14px;
    border:1px solid rgba(255,255,255,0.08);
    background: linear-gradient(150deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
    backdrop-filter: blur(14px);
    transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
    animation: rise .45s ease both;
}
.pcard:hover {transform: translateY(-3px);}
.pcard.win {border-color: rgba(255,255,255,0.22);}
@keyframes rise {from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:none;}}

.pcard-top {display:flex; align-items:center; justify-content:space-between; gap:12px;}
.pcard-id {display:flex; align-items:center; gap:10px;}
.pcard-icon {font-size:1.15rem; line-height:1;}
.pcard-name {font-weight:600; font-size:1.02rem; letter-spacing:.01em;}
.pcard-sub {font-size:.74rem; color:#7d86a1; margin-top:2px;}
.pcard-digit {
    font-family:'JetBrains Mono', monospace; font-weight:600;
    font-size:2.9rem; line-height:1; min-width:56px; text-align:right;
}
.crown {
    font-size:.62rem; font-family:'JetBrains Mono',monospace; letter-spacing:.14em;
    text-transform:uppercase; padding:3px 9px; border-radius:999px; margin-left:8px;
    border:1px solid rgba(255,255,255,0.2); color:#e8ecf8;
}
.bar-track {
    height:7px; border-radius:999px; background:rgba(255,255,255,0.07);
    margin-top:14px; overflow:hidden;
}
.bar-fill {height:100%; border-radius:999px; animation: grow .7s cubic-bezier(.22,1,.36,1) both;}
@keyframes grow {from{width:0 !important;} }
.bar-meta {
    display:flex; justify-content:space-between; margin-top:7px;
    font-family:'JetBrains Mono',monospace; font-size:.7rem; color:#7d86a1;
}

/* ---------------- consensus banner ---------------- */
.consensus {
    border-radius:16px; padding:14px 18px; margin-bottom:16px;
    border:1px solid rgba(255,255,255,0.1); backdrop-filter:blur(12px);
    display:flex; align-items:center; gap:12px;
    animation: rise .4s ease both;
}
.consensus .big {font-size:1.6rem; line-height:1;}
.consensus .t1 {font-weight:600; font-size:.98rem;}
.consensus .t2 {font-size:.8rem; color:#9aa3bd; margin-top:2px;}

/* ---------------- probability grid ---------------- */
.prow {display:flex; align-items:center; gap:10px; margin-bottom:5px;}
.prow .d {
    font-family:'JetBrains Mono',monospace; font-size:.76rem; color:#8b93ad;
    width:14px; text-align:center;
}
.prow .track {flex:1; height:9px; border-radius:999px; background:rgba(255,255,255,0.06); overflow:hidden;}
.prow .fill {height:100%; border-radius:999px; animation: grow .6s cubic-bezier(.22,1,.36,1) both;}
.prow .v {
    font-family:'JetBrains Mono',monospace; font-size:.68rem; color:#7d86a1;
    width:46px; text-align:right;
}

/* ---------------- empty state ---------------- */
.empty {
    border:1px dashed rgba(255,255,255,0.16); border-radius:18px;
    padding:46px 24px; text-align:center; color:#8b93ad;
    background: rgba(255,255,255,0.02);
}
.empty .ico {font-size:2.4rem; display:block; margin-bottom:10px; opacity:.75;}
.empty b {color:#e8ecf8; display:block; margin-bottom:6px; font-size:1.02rem;}

/* ---------------- scoreboard ---------------- */
.score {
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 14px; border-radius:12px; margin-bottom:8px;
    border:1px solid rgba(255,255,255,0.07); background:rgba(255,255,255,0.03);
}
.score .nm {font-size:.88rem; font-weight:500;}
.score .rt {font-family:'JetBrains Mono',monospace; font-size:.88rem;}

/* ---------------- streamlit widget overrides ---------------- */
.stButton > button {
    width:100%; border-radius:12px; font-weight:600; font-size:.88rem;
    border:1px solid rgba(255,255,255,0.14);
    background: linear-gradient(135deg, rgba(34,211,238,0.16), rgba(168,85,247,0.16));
    color:#e8ecf8; transition: all .18s ease; padding:.5rem 1rem;
}
.stButton > button:hover {
    border-color: rgba(34,211,238,0.6);
    box-shadow: 0 0 22px rgba(34,211,238,0.22);
    transform: translateY(-1px); color:#ffffff;
}
div[data-testid="stExpander"] {
    border:1px solid rgba(255,255,255,0.08) !important; border-radius:14px !important;
    background: rgba(255,255,255,0.03) !important; margin-bottom:10px;
}
div[data-testid="stExpander"] summary {font-weight:600; font-size:.9rem;}
.stSlider label, .stSelectbox label, .stCheckbox label {
    font-size:.8rem !important; color:#9aa3bd !important;
}
hr {border-color: rgba(255,255,255,0.08);}
.canvas-wrap {display:flex; justify-content:center;}
.tinyhint {font-size:.74rem; color:#6f7893; text-align:center; margin-top:8px;}
.foot {text-align:center; color:#6f7893; font-size:.78rem; padding:26px 0 6px 0;}
.foot a {color:#22d3ee; text-decoration:none;}
</style>
""",
        unsafe_allow_html=True,
    )


inject_css()


# --------------------------------------------------------------------------- #
# Model loading + inference
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "scores" not in st.session_state:
    st.session_state.scores = {n: {"correct": 0, "total": 0} for n in MODEL_META}


def clear_canvas():
    st.session_state.canvas_key += 1


def reset_scores():
    st.session_state.scores = {n: {"correct": 0, "total": 0} for n in MODEL_META}


# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #

st.markdown(
    '<div class="hero">'
    '<div class="hero-badge">MNIST · Live Inference</div>'
    "<h1>One digit, three minds.</h1>"
    "<p>Draw a number and watch a single-layer <b>Perceptron</b>, a fully-connected "
    "<b>ANN</b> and a <b>CNN</b> argue about what you meant — complete with the "
    "confidence each one is willing to stake on it.</p>"
    "</div>",
    unsafe_allow_html=True,
)

if not os.path.exists(os.path.join(MODEL_DIR, "cnn.keras")):
    st.error(
        "No trained models found in `saved_models/`. Run `python src/train.py` "
        "locally first, then commit the `.keras` files (or point this app at them)."
    )
    st.stop()

models = load_models()

# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown('<div class="section-label">Brush</div>', unsafe_allow_html=True)
    stroke_width = st.slider("Stroke width", 8, 40, 18, 1)
    canvas_size = st.select_slider("Canvas size", options=[240, 280, 320, 360], value=280)
    st.button("Clear canvas", on_click=clear_canvas)

    st.markdown("---")
    st.markdown('<div class="section-label">Display</div>', unsafe_allow_html=True)
    show_probs = st.checkbox("Show probability distributions", value=True)
    show_input = st.checkbox("Show what the models see (28×28)", value=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Scoreboard</div>', unsafe_allow_html=True)
    st.caption("Tell the app what you actually drew to keep score.")
    for name, meta in MODEL_META.items():
        s = st.session_state.scores[name]
        pct = f"{s['correct'] / s['total'] * 100:.0f}%" if s["total"] else "—"
        st.markdown(
            f'<div class="score"><span class="nm" style="color:{meta["color"]}">'
            f'{meta["icon"]} {name}</span>'
            f'<span class="rt">{s["correct"]}/{s["total"]} · {pct}</span></div>',
            unsafe_allow_html=True,
        )
    st.button("Reset scoreboard", on_click=reset_scores)

# --------------------------------------------------------------------------- #
# Main layout
# --------------------------------------------------------------------------- #

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown('<div class="section-label">Draw a digit (0–9)</div>', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=stroke_width,
        stroke_color="#ffffff",
        background_color="#000000",
        height=canvas_size,
        width=canvas_size,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )
    st.markdown(
        '<div class="tinyhint">Predictions update the moment you lift the pen · '
        "big, centred strokes work best</div>",
        unsafe_allow_html=True,
    )

has_drawing = (
    canvas_result.image_data is not None
    and canvas_result.image_data[:, :, :3].sum() > 0
)

arr = None
results = None
if has_drawing:
    arr = preprocess(canvas_result.image_data)
    results = predict_all(models, arr)

with left:
    if has_drawing and show_input:
        st.markdown(
            '<div class="section-label" style="margin-top:18px">Model input · 28 × 28</div>',
            unsafe_allow_html=True,
        )
        preview = Image.fromarray((arr * 255).astype("uint8"), mode="L").resize(
            (168, 168), Image.NEAREST
        )
        st.image(preview, caption="Downsampled and normalised exactly as MNIST was")

with right:
    st.markdown('<div class="section-label">Predictions</div>', unsafe_allow_html=True)

    if not has_drawing:
        st.markdown(
            '<div class="empty"><span class="ico">✎</span>'
            "<b>Nothing on the canvas yet</b>"
            "Sketch a digit on the left and all three models will weigh in instantly."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        preds = {n: r["prediction"] for n, r in results.items()}
        winner = max(results, key=lambda n: results[n]["confidence"])
        unique = sorted(set(preds.values()))

        if len(unique) == 1:
            c = "#34d399"
            banner = (
                f'<div class="consensus" style="background:rgba(52,211,153,0.10);'
                f'border-color:rgba(52,211,153,0.35)">'
                f'<span class="big" style="color:{c}">✓</span><div>'
                f'<div class="t1">All three agree — it\'s a {unique[0]}</div>'
                f'<div class="t2">Unanimous vote across Perceptron, ANN and CNN.</div>'
                f"</div></div>"
            )
        else:
            c = "#fbbf24"
            split = " vs ".join(
                f"{n} says {preds[n]}" for n in ["Perceptron", "ANN", "CNN"]
            )
            banner = (
                f'<div class="consensus" style="background:rgba(251,191,36,0.10);'
                f'border-color:rgba(251,191,36,0.35)">'
                f'<span class="big" style="color:{c}">⚡</span><div>'
                f'<div class="t1">The models disagree</div>'
                f'<div class="t2">{split}. This is exactly where architecture starts to matter.</div>'
                f"</div></div>"
            )
        st.markdown(banner, unsafe_allow_html=True)

        for name, res in results.items():
            meta = MODEL_META[name]
            pct = res["confidence"] * 100
            is_win = name == winner
            glow = (
                f"box-shadow:0 0 0 1px rgba({meta['glow']},0.35), "
                f"0 12px 38px rgba({meta['glow']},0.20);"
                if is_win
                else ""
            )
            crown = (
                f'<span class="crown" style="border-color:rgba({meta["glow"]},0.5);'
                f'background:rgba({meta["glow"]},0.14);color:{meta["color"]}">'
                f"most confident</span>"
                if is_win
                else ""
            )
            st.markdown(
                f'<div class="pcard{" win" if is_win else ""}" style="{glow}">'
                f'<div class="pcard-top"><div class="pcard-id">'
                f'<span class="pcard-icon" style="color:{meta["color"]}">{meta["icon"]}</span>'
                f'<div><div class="pcard-name">{name}{crown}</div>'
                f'<div class="pcard-sub">{meta["tagline"]}</div></div></div>'
                f'<div class="pcard-digit" style="color:{meta["color"]};'
                f'text-shadow:0 0 26px rgba({meta["glow"]},0.55)">{res["prediction"]}</div>'
                f'</div><div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;'
                f'background:linear-gradient(90deg,rgba({meta["glow"]},0.45),{meta["color"]});'
                f'box-shadow:0 0 14px rgba({meta["glow"]},0.6)"></div></div>'
                f'<div class="bar-meta"><span>confidence</span>'
                f"<span>{pct:.1f}%</span></div></div>",
                unsafe_allow_html=True,
            )

        # ---- scoreboard entry ----
        with st.expander("Was it right? Record the answer", expanded=False):
            c1, c2 = st.columns([1, 1])
            with c1:
                truth = st.selectbox("The digit I actually drew", list(range(10)), key="truth")
            with c2:
                st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
                if st.button("Record round"):
                    for n, r in results.items():
                        st.session_state.scores[n]["total"] += 1
                        if r["prediction"] == int(truth):
                            st.session_state.scores[n]["correct"] += 1
                    st.success("Logged — see the scoreboard in the sidebar.")

        if show_probs:
            st.markdown(
                '<div class="section-label" style="margin-top:20px">Probability distribution</div>',
                unsafe_allow_html=True,
            )
            tabs = st.tabs(list(results.keys()))
            for tab, (name, res) in zip(tabs, results.items()):
                meta = MODEL_META[name]
                with tab:
                    rows = ""
                    top = int(np.argmax(res["probs"]))
                    for d, p in enumerate(res["probs"]):
                        w = max(float(p) * 100, 0.6)
                        color = meta["color"] if d == top else "rgba(255,255,255,0.22)"
                        shadow = (
                            f"box-shadow:0 0 12px rgba({meta['glow']},0.55);"
                            if d == top
                            else ""
                        )
                        dcol = meta["color"] if d == top else "#8b93ad"
                        rows += (
                            f'<div class="prow"><span class="d" style="color:{dcol}">{d}</span>'
                            f'<span class="track"><span class="fill" style="width:{w:.2f}%;'
                            f'background:{color};{shadow}"></span></span>'
                            f'<span class="v">{float(p)*100:.2f}%</span></div>'
                        )
                    st.markdown(f'<div class="panel">{rows}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Model reference
# --------------------------------------------------------------------------- #

st.markdown(
    '<div class="section-label" style="margin-top:26px">Under the hood</div>',
    unsafe_allow_html=True,
)
cols = st.columns(3, gap="medium")
for col, (name, model) in zip(cols, zip(MODEL_META.keys(), models)):
    meta = MODEL_META[name]
    try:
        params = f"{model.count_params():,}"
    except Exception:
        params = "—"
    with col:
        with st.expander(f"{meta['icon']}  {name}", expanded=False):
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:.72rem;'
                f'color:#7d86a1;letter-spacing:.12em;text-transform:uppercase">'
                f"{params} params · {meta['accuracy']} test acc</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='margin-top:10px'>"
                + "".join(
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:.76rem;'
                    f'color:#c7cfe4;padding:5px 10px;margin-bottom:5px;border-radius:8px;'
                    f'background:rgba({meta["glow"]},0.08);'
                    f'border-left:2px solid {meta["color"]}">{layer}</div>'
                    for layer in meta["arch"]
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-size:.82rem;color:#9aa3bd;line-height:1.55;margin-top:10px'>"
                f"{meta['note']}</p>",
                unsafe_allow_html=True,
            )

st.markdown(
    '<div class="foot">Built with TensorFlow / Keras + Streamlit · '
    'Source &amp; training code on '
    '<a href="https://github.com/himanshuydv9214/Number_argue">GitHub</a></div>',
    unsafe_allow_html=True,
)
