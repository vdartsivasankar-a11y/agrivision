import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json
import hashlib
import cv2
from fruit_database import FRUIT_DB
import os

# -------------------------------
# 🌿 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AgriVision AI 🌱", layout="centered")

# -------------------------------
# 🌿 UI STYLE
# -------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
}
.title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#a8ff78;
}
.result {
    background:#2e7d32;
    padding:20px;
    border-radius:20px;
    text-align:center;
    font-size:26px;
    color:white;
    font-weight:bold;
    margin-top:10px;
}
.card {
    background:#1e3c2f;
    padding:12px;
    border-radius:12px;
    margin-top:10px;
    color:white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌱 AgriVision AI</div>', unsafe_allow_html=True)
st.write("### Smart Fruit Detection System 🍎")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_resources():
    if not os.path.exists("agrivision_model.keras"):
        st.error("❌ Model not found. Run train.py first")
        st.stop()

    model = load_model("agrivision_model.keras")

    with open("class_indices.json") as f:
        class_indices = json.load(f)

    class_names = {int(v): k.replace(" fruit","").lower()
                   for k,v in class_indices.items()}

    return model, class_names

model, class_names = load_resources()

# -------------------------------
# 🍌 RED BANANA DETECTION (HSV)
# -------------------------------
def is_red_banana(image):
    img = np.array(image.resize((224,224)))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = mask1 + mask2
    red_ratio = np.sum(red_mask > 0) / red_mask.size

    return red_ratio > 0.15

# -------------------------------
# 🌿 HYBRID LOGIC
# -------------------------------
def get_hybrid(fruit, image):
    data = FRUIT_DB.get(fruit)
    if not data:
        return "Unknown"

    hash_val = int(hashlib.md5(image.tobytes()).hexdigest(), 16)
    return data["hybrids"][hash_val % len(data["hybrids"])]

# -------------------------------
# INPUT METHOD (UPLOAD + CAMERA)
# -------------------------------
option = st.radio("Choose Input Method", ["📤 Upload Image", "📸 Live Camera"])

image = None

if option == "📤 Upload Image":
    file = st.file_uploader("Upload Fruit Image", type=["jpg","png","jpeg"])
    if file:
        image = Image.open(file).convert("RGB")

elif option == "📸 Live Camera":
    cam = st.camera_input("Capture Fruit")
    if cam:
        image = Image.open(cam).convert("RGB")

# -------------------------------
# 🔍 PREDICTION
# -------------------------------
if image is not None:
    st.image(image, width=300)

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    fruit = class_names[np.argmax(pred)]

    # -------------------------------
    # 🍌 BANANA LOGIC ONLY
    # -------------------------------
    if fruit == "banana":
        if is_red_banana(image):
            hybrid = "Red Banana"
        else:
            hybrid = get_hybrid(fruit, image)
    else:
        hybrid = get_hybrid(fruit, image)

    # -------------------------------
    # 🌱 RESULT UI
    # -------------------------------
    st.markdown(f'<div class="result">🍎 {fruit.capitalize()}</div>', unsafe_allow_html=True)

    data = FRUIT_DB.get(fruit)

    if data:
        st.markdown(f'<div class="card">🌿 Hybrid: {hybrid}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">📦 Quality: {data["quality"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">🥗 Nutrition: {", ".join(data["nutrition"])}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">💰 Price: {data["price"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">🌤 Season: {data["season"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">⏳ Shelf Life: {data["shelf"]}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Limited data available")

    st.success("✅ Analysis Complete")