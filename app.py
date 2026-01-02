import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# App Config

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

st.title("Face Mask Detection")
st.write("Upload an image to check whether the person is wearing a mask.")


# Load Model

MODEL_PATH = "mask_detection_model.keras"
model = load_model(MODEL_PATH)


# Image Upload

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# Prediction Logic

if uploaded_file is not None:
    # Read image using PIL
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Resize to model input size
    image_resized = cv2.resize(image_np, (128, 128))

    # Scale pixel values
    image_scaled = image_resized / 255.0

    # Reshape for model
    image_input = np.reshape(image_scaled, (1, 128, 128, 3))

    # Predict
    prediction = model.predict(image_input)
    pred_label = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    # Display result
    if pred_label == 1:
        st.success(f" Mask Detected ({confidence:.2f}%)")
    else:
        st.error(f"No Mask Detected ({confidence:.2f}%)")
