import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Deepfake Detector", layout="centered")

# Load the trained model
model = load_model("model/mesonet_trained.h5")

# Streamlit UI
st.title("🕵️‍♂️ Deepfake Detection App")
st.write("Upload an image and the model will predict whether it's REAL or FAKE.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    image = image.resize((256, 256))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)[0][0]

    label = "REAL" if prediction > 0.5 else "FAKE"
    confidence = 1 - prediction if prediction < 0.5 else prediction

    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2f}`")