import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. APPLICATION SETUP ---
st.set_page_config(
    page_title="GPR-X Detection Suite", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimalist Custom CSS for an official corporate look
st.markdown("""
    <style>
    /* Remove default Streamlit top decoration line */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global styling overrides */
    .reportview-container {
        background: #FAFAFA;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #0F172A;
        font-weight: 600;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. OFFICIAL ENTERPRISE HEADER ---
st.markdown("""
    <div style="padding-bottom: 20px; border-bottom: 2px solid #E2E8F0; margin-bottom: 35px;">
        <h1 style="color:#0F172A; margin:0; font-size:2.2rem; font-weight:700; letter-spacing:-0.5px;">GPR-X Subsurface Analysis Suite</h1>
        <p style="color:#64748B; margin:5px 0 0 0; font-size:1rem; font-weight:400;">Automated Computer Vision Target Identification</p>
    </div>
""", unsafe_allow_html=True)

# --- 3. MODEL CORE ENGINE ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Engine Core Error: 'best.pt' missing from repository root.")

# --- 4. INTEGRATED DATA ROUTING INTERFACE ---
mode = st.radio(
    "Data Source Selection:",
    ["Interactive Profiles (Exhibition Demo)", "External B-Scan Upload"],
    horizontal=True
)

img = None

if mode == "External B-Scan Upload":
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
else:
    # 7 exact file names matching your local folder
    sample_options = {
        "Profile 01: Lab Test — Buried Cavity": "samples/exp_cavity.png",
        "Profile 02: Lab Test — Metal Pipe": "samples/exp_metal_pipe.png",
        "Profile 03: Lab Test — Concrete Block": "samples/exp_concrete.png",
        "Profile 04: Lab Test — Pipe & Cavity Combo": "samples/exp_pipe_cavity.png",
        "Profile 05: Lab Test — Full Structural Mix": "samples/exp_all.png",
        "Profile 06: Field Scan — Manhole Cover": "samples/real_manhole.jpg",
        "Profile 07: Field Scan — Real-World Cavity": "samples/real_cavity.JPG"
    }
    
    selected_sample = st.selectbox("", list(sample_options.keys()), label_visibility="collapsed")
    sample_path = sample_options[selected_sample]
    
    if os.path.exists(sample_path):
        img = Image.open(sample_path).convert("RGB")
    else:
        st.warning(f"System File Error: Target array missing at `{sample_path}`.")

# --- 5. COMPACT INTERFERENCE & DISPLAY ENGINE ---
if img is not None:
    img_array = np.array(img)

    with st.spinner("Executing localization algorithms..."):
        results = model.predict(source=img_array, conf=0.25)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        num_detections = len(results[0].boxes)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sharp Side-by-Side Presentation
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<p style='color:#64748B; font-weight:600; margin-bottom:10px;'>RAW RADAR INPUT MATRIX</p>", unsafe_allow_html=True)
        st.image(img, use_container_width=True)
    with col2:
        st.markdown("<p style='color:#0F172A; font-weight:600; margin-bottom:10px;'>AI ANOMALY MAPPING OUTPUT</p>", unsafe_allow_html=True)
        st.image(res_rgb, use_container_width=True)

    # --- 6. METRICS OVERVIEW ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    m_col1, m_col2 = st.columns([1, 3])
    with m_col1:
        st.metric(label="ISOLATED OBJECTS", value=f"{num_detections}")
    with m_col2:
        st.markdown("<p style='color:#64748B; font-weight:600; margin:0;'>DIAGNOSTIC TELEMETRY REPORT</p>", unsafe_allow_html=True)
        if num_detections > 0:
            st.success(f"Analysis cycle complete. {num_detections} distinct targets registered above baseline confidence index.")
        else:
            st.info("Analysis cycle complete. Frame verification successful. No structural anomalies registered.")
            
else:
    if mode == "External B-Scan Upload":
        st.markdown("<br><p style='color:#94A3B8;'>Awaiting system image load parameters...</p>", unsafe_allow_html=True)
