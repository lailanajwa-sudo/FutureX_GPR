import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="GPR-X Detector", layout="wide")

st.title("🛰️ GPR-X Detector")
st.write("AI Based Subsurface GPR Detection")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Model 'best.pt' not found. Please upload it to your GitHub repository.")

# --- 3. EXHIBITION INTERACTIVE WORKFLOW ---
st.divider()

# Creating two options to split Upload vs Demo Mode
mode = st.radio(
    "Choose how you want to try GPR-X:",
    ["Use Exhibition Sample Data (Interactive Demo)", "Upload My Own Radargram"],
    horizontal=True
)

img = None

if mode == "Upload My Own Radargram":
    uploaded_file = st.file_uploader("Upload Radargram B-Scan", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
else:
    st.markdown("### 🎯 Select a sample radargram dataset below to test the AI:")
    
    # Dictionary matching your 7 new short filenames perfectly
    sample_options = {
        "1. Lab Test: Buried Cavity": "samples/exp_cavity.png",
        "2. Lab Test: Metal Pipe": "samples/exp_metal_pipe.png",
        "3. Lab Test: Concrete Block": "samples/exp_concrete.png",
        "4. Lab Test: Pipe & Cavity Combo": "samples/exp_pipe_cavity.png",
        "5. Lab Test: Full Mix (Concrete, Pipe, Cavity)": "samples/exp_all.png",
        "6. Field Scan: Manhole Cover": "samples/real_manhole.jpg",
        "7. Field Scan: Real-World Sinkhole / Cavity": "samples/real_cavity.JPG"
    }
    
    selected_sample = st.selectbox("Pick a scenario to detect:", list(sample_options.keys()))
    sample_path = sample_options[selected_sample]
    
    # Check if file exists, then load it
    if os.path.exists(sample_path):
        img = Image.open(sample_path).convert("RGB")
        st.success(f"Loaded {selected_sample} successfully!")
    else:
        st.warning(f"Demo file not found at `{sample_path}`. Make sure it is saved in your samples folder.")

# --- 4. PROCESSING & DETECTION LAYER ---
if img is not None:
    img_array = np.array(img)

    with st.spinner("AI analyzing radargram..."):
        # Run AI Inference
        results = model.predict(source=img_array, conf=0.25)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Count number of boxes detected
        num_detections = len(results[0].boxes)

    # --- 5. DISPLAY RESULTS ---
    st.markdown("### 📊 Classification Results")
    
    # Simple Side-by-Side layout
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Input Scan**")
        st.image(img, use_container_width=True)
    with col2:
        st.write("**AI Predictions & Hyperbola Fixes**")
        st.image(res_rgb, use_container_width=True)

    # --- 6. NUMERICAL RESULTS ---
    st.divider()
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("Summary")
        st.write(f"✅ **Total Anomalies Identified:** {num_detections}")
        st.write(f"The model successfully classified **{num_detections}** targets in this scan.")
    
    with res_col2:
        st.info("💡 **Exhibition Tip:** Look at how the bounding boxes hug the peak apexes of the hyperbolas! This represents real-time feature extraction mapping.")

else:
    if mode == "Upload My Own Radargram":
        st.info("Please upload a radargram to begin classification.")
