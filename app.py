import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. PAGE SETUP (OFFICIAL & MINIMALIST) ---
st.set_page_config(
    page_title="GPR-X", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide default Streamlit clutter for a clean corporate look
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0B0F19 !important;
    }
    /* Clean custom card layout for summary */
    .report-card {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 20px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Minimalist Corporate Title Header
st.markdown("""
    <div style="padding-bottom: 15px; border-bottom: 1px solid #1E293B; margin-bottom: 30px;">
        <h1 style="color:#FFFFFF; margin:0; font-size:2rem; font-weight:700; letter-spacing:-0.5px;">AI-Based Subsurface GPR Detection</h1>
        <p style="color:#64748B; margin:5px 0 0 0; font-size:0.95rem;">Automated Computer Vision Target Identification</p>
    </div>
""", unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Engine Core Error: 'best.pt' missing from repository root.")

# --- 3. INTERACTIVE WORKFLOW ---
mode = st.radio(
    "INPUT CONTROL CHANNEL",
    ["Use Exhibition Sample Data (Interactive Demo)", "Upload My Own Radargram"],
    label_visibility="collapsed",
    horizontal=True
)

img = None

if mode == "Upload My Own Radargram":
    uploaded_file = st.file_uploader("Upload Radargram B-Scan", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
else:
    # Set the first option to None to block the automatic preview of Sample 1
    sample_options = {
        "-- Select a Scenario --": None,
        "1. Lab Test: Buried Cavity": "samples/exp_cavity.png",
        "2. Lab Test: Metal Pipe": "samples/exp_metal_pipe.png",
        "3. Lab Test: Concrete Block": "samples/exp_concrete.png",
        "4. Lab Test: Pipe & Cavity": "samples/exp_pipe_cavity.png",
        "5. Lab Test: Concrete, Pipe, Cavity": "samples/exp_all.png",
        "6. Field Scan: Manhole Cover": "samples/real_manhole.jpg",
        "7. Field Scan: Real-World Sinkhole / Cavity": "samples/real_cavity.JPG"
    }
    
    selected_sample = st.selectbox("Pick a scenario to detect:", list(sample_options.keys()))
    sample_path = sample_options[selected_sample]
    
    # Only read the file if the user picks a real dataset profile
    if sample_path is not None:
        if os.path.exists(sample_path):
            img = Image.open(sample_path).convert("RGB")
        else:
            st.error(f"File not found at `{sample_path}`. Please verify your samples directory.")

# --- 4. PROCESSING & DETECTION LAYER ---
# The analysis and screen rendering only activate if an image is actively loaded
if img is not None:
    img_array = np.array(img)

    with st.spinner("AI analyzing radargram matrices..."):
        # Run AI Inference
        results = model.predict(source=img_array, conf=0.25, verbose=False)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        num_detections = len(results[0].boxes)

    st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Analysis Output")
    
    # Sharp Side-by-Side Presentation layout
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.write("**Original Input Scan**")
        st.image(img, use_container_width=True)
    with col2:
        st.write("**AI Predictions & Diagnostics**")
        st.image(res_rgb, use_container_width=True)

    # --- 5. NUMERICAL RESULTS ---
    st.markdown("""
        <div class="report-card">
            <h4 style="color:#94A3B8; margin:0 0 10px 0; font-size:0.85rem; letter-spacing:1px;">SYSTEM TELEMETRY SUMMARY</h4>
            <p style="color:#FFFFFF; margin:0; font-size:1.1rem;">
                ✅ <b>Total Anomalies Identified:</b> <span style="color:#38BDF8; font-family:monospace; font-size:1.3rem;">'""" + f"{num_detections:02d}" + """'</span> targets detected.
            </p>
            <p style="color:#94A3B8; margin:5px 0 0 0; font-size:0.9rem;">
                The deep learning model has locked bounding frames precisely onto the detected hyperbolic apexes.
            </p>
        </div>
    """, unsafe_allow_html=True)

else:
    # Landing state layout when users haven't selected anything yet
    st.markdown("<br>", unsafe_allow_html=True)
    if mode == "Upload My Own Radargram":
        st.info("Awaiting B-Scan file upload to initialize neural network layer...")
    else:
        st.info("Awaiting scenario selection. Please select a radar profile from the dropdown menu above.")
