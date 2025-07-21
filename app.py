import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import json

# --- Utility: Load class advice/status from JSON ---
@st.cache_data
def load_class_messages():
    with open("class_messages.json", "r") as f:
        return json.load(f)
class_messages = load_class_messages()

# --- Utility: Get model labels (Update to match your training) ---
CLASS_NAMES = [
    "Apple___black_rot", "Apple___healthy", "Apple___rust", "Apple___scab",
    "Blueberry___healthy", "Cherry___healthy", "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot", "Corn___Common_rust",
    "Corn___healthy", "Corn___Northern_Leaf_Blight", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper_bell___Bacterial_spot", "Pepper_bell___healthy", "Potato___Early_blight",
    "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy",
    "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___healthy",
    "Strawberry___Leaf_scorch", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites_Two_spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# --- Helper: Load the trained CNN only once ---
@st.cache_resource
def load_cnn():
    return load_model("cnn_model.h5")

model = load_cnn()

# --- App UI ---
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌱", layout="centered")
st.title("🌿 Plant Disease Classifier (CNN)")
st.write(
    "Upload a leaf image. The app will predict the disease class, show severity, and recommend treatment."
)

uploaded = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", width=256)

    # --- Preprocess ---
    img_size = (128, 128)  # Update if your CNN expects a different size
    img_arr = image.resize(img_size)
    img_arr = np.array(img_arr) / 255.0
    img_input = np.expand_dims(img_arr, axis=0)

    # --- Predict ---
    preds = model.predict(img_input)
    class_idx = np.argmax(preds)
    pred_class = CLASS_NAMES[class_idx]
    confidence = float(np.max(preds))

    # --- Retrieve class status and message ---
    info = class_messages.get(pred_class, {
        "status": "Unknown",
        "message": "No specific guidance available for this class."
    })

    st.success(f"**Prediction:** {pred_class}")
    st.write(f"**Status:** `{info['status']}`")
    st.info(f"**Recommended Action:** {info['message']}")
    st.write(f"**Model confidence:** `{confidence:.2%}`")

    # Optionally: Show raw probabilities for research/debugging
    #st.write("All class probabilities:", preds)
else:
    st.info("Please upload a JPG/PNG image of a leaf.")
