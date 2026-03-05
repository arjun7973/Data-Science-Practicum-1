import json
import cv2
import numpy as np
from pathlib import Path

import streamlit as st
from tensorflow import keras
import tensorflow as tf

# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
APP_DIR   = Path(__file__).parent
MODEL_PATH = APP_DIR / "model" / "model.keras"
INFO_PATH  = APP_DIR / "model" / "class_info.json"

RISK_COLOR = {"Benign": "green", "Malignant": "red", "Precancerous": "orange"}

# ──────────────────────────────────────────
# Load model + class metadata (cached)
# ──────────────────────────────────────────
@st.cache_resource
def load_model():
    return keras.models.load_model(str(MODEL_PATH))

@st.cache_data
def load_class_info():
    with open(INFO_PATH) as f:
        return json.load(f)

# ──────────────────────────────────────────
# Preprocessing (mirrors training pipeline)
# ──────────────────────────────────────────
def preprocess(img_bgr, size=256):
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    img = cv2.inpaint(img, hair_mask, 1, cv2.INPAINT_TELEA)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_inference(model, class_info, img_bgr):
    processed = preprocess(img_bgr)
    arr = tf.cast(processed, tf.float32)[tf.newaxis, ...]
    probs = model(arr, training=False).numpy()[0]
    return {cls: float(p) for cls, p in zip(class_info["class_names"], probs)}, processed


# ──────────────────────────────────────────
# UI
# ──────────────────────────────────────────
st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")
st.title("Skin Lesion Classification")
st.caption("ISIC 2019 · EfficientNetB2 · 8 classes")

uploaded = st.file_uploader("Upload a dermoscopic image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    model      = load_model()
    class_info = load_class_info()

    probs, processed = run_inference(model, class_info, img_bgr)

    top_cls  = max(probs, key=probs.get)
    top_prob = probs[top_cls]
    risk     = class_info["clinical_risk"][top_cls]
    desc     = class_info["descriptions"][top_cls]

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    with col2:
        st.image(processed, caption="Preprocessed", use_container_width=True)

    st.markdown("---")

    color = RISK_COLOR.get(risk, "gray")
    st.markdown(f"### Prediction: **{top_cls}** &nbsp; :{color}[{risk}]")
    st.write(desc)
    st.metric("Confidence", f"{top_prob:.1%}")

    st.markdown("#### Class Probabilities")
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
    for cls, p in sorted_probs:
        st.progress(p, text=f"{cls}: {p:.2%}")

else:
    st.info("Upload an image to get started.")
