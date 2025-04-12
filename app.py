import streamlit as st
from predict_disease import predict_disease
import os
import shutil

# Page configuration
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")

# Title and instructions
st.title("🌿 Plant Disease Detector")
st.markdown("Upload an image of a **plant leaf** to detect the disease and get suggestions for treatment.")

# Upload section
uploaded_file = st.file_uploader("📁 Choose a plant leaf image", type=["jpg", "jpeg", "png"])

# Prepare upload directory
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

if uploaded_file is not None:
    # Save uploaded file
    image_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(image_path, caption="🖼️ Uploaded Leaf Image", use_column_width=True)

    # Predict
    with st.spinner("🔍 Analyzing the image..."):
        try:
            prediction, suggestion = predict_disease(image_path)
            st.success(f"🌱 **Prediction:** {prediction}")
            st.info(f"💡 **Suggestion:** {suggestion}")
        except Exception as e:
            st.error("⚠️ Something went wrong during prediction.")
            st.exception(e)

    # Optional: Clean uploads folder
    if st.checkbox("🧹 Delete uploaded image after prediction"):
        try:
            os.remove(image_path)
            st.success("Uploaded image deleted successfully.")
        except Exception as e:
            st.warning(f"Could not delete image: {e}")
