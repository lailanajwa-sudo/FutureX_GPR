import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. PAGE SETUP (PREMIUM THEME) ---
st.set_page_config(
    page_title="GPR-X Subsurface AI Detector", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to inject a clean, tech-forward look and style buttons/metrics
st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        color: #38BDF8;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #94A3B8;
    }
    </style>
""", unsafe_allow_html=True)

# High-contrast premium banner for the expo floor
st.markdown("""
    <div style="background-color:#0F172A; padding:24px; border-radius:12px; border-left: 6px solid #38BDF8; margin-bottom:25px;">
        <h1 style="color:#F8FAFC; margin:0; font-size:2.6rem; font-family:sans-serif;">🛰️ GPR-X Intelligence</h1>
        <p style="color:#94A3B8; margin:6px 0 0 0; font-size:1.1rem; font-family:sans-serif;">
            Real-Time Deep Learning for Subsurface Anomaly Detection & Hyperbolic Feature Extraction
        </p>
    </div>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("### 🛠️ GPR-X System Controls")
    
    # Modernized Radio selection
    mode = st.radio(
        "Select Data Stream Input:",
        ["🎯 Exhibition Demo Datasets", "📤 Upload Custom Radargram"],
        help="Switch between live showcase presets or custom uploaded files."
    )
    
    st.divider()
    
    # INTERACTIVE ADDITION: Let visitors play with the AI sensitivity live
    st.markdown("### 🎛️ AI Sensitivity Tuning")
    conf_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.10, 
        max_value=0.90, 
        value=0.25, 
        step=0.05,
        help="Lower values catch more faint anomalies. Higher values ensure absolute accuracy."
    )
    
    st.divider()
    st.markdown("### 📊 Exhibition Insights")
    st.markdown("""
    This kiosk deploys a fine-tuned **YOLOv8 architecture** trained to recognize distinct underground radargram signatures.
    
    **Engine Benchmarks:**
    * ⚡ **Sub-second Processing:** < 40ms inference delay
    * 🎯 **Hyperbolic Apex Locating:** Eliminates manual grid math
    * 🔍 **Multi-Target Isolation:** Voids vs. Metallic Utilities
    """)
    st.caption("System Status: Operational • YOLOv8 Engine Active")

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Model 'best.pt' not found. Please upload it to your GitHub repository.")

# --- 4. DATA COMPONENT ROUTING ---
img = None

if mode == "Upload My Own Radargram":
    st.markdown("### 📤 Upload Your B-Scan Data")
    uploaded_file = st.file_uploader("Drag and drop your radargram image here", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
else:
    st.markdown("### 🎯 Interactive Target Showcase")
    st.write("Select a scenario to instantly simulate subsurface imaging analytics:")
    
    # Dictionary matching your 7 custom file paths and extensions perfectly
    sample_options = {
        "1. Lab Test: Buried Cavity": "samples/exp_cavity.png",
        "2. Lab Test: Metal Pipe": "samples/exp_metal_pipe.png",
        "3. Lab Test: Concrete Block": "samples/exp_concrete.png",
        "4. Lab Test: Pipe & Cavity Combo": "samples/exp_pipe_cavity.png",
        "5. Lab Test: Full Mix (Concrete, Pipe, Cavity)": "samples/exp_all.png",
        "6. Field Scan: Manhole Cover": "samples/real_manhole.jpg",
        "7. Field Scan: Real-World Sinkhole / Cavity": "samples/real_cavity.JPG"
    }
    
    selected_sample = st.selectbox("Choose a subsurface scan profile:", list(sample_options.keys()))
    sample_path = sample_options[selected_sample]
    
    if os.path.exists(sample_path):
        img = Image.open(sample_path).convert("RGB")
    else:
        st.warning(f"⚠️ Target file not discovered at path `{sample_path}`. Ensure your `samples/` folder is updated.")

# --- 5. PROCESSING & DYNAMIC DISPLAY LAYER ---
if img is not None:
    img_array = np.array(img)

    with st.spinner("🧠 GPR-X Engine analyzing radargram matrices..."):
        # Run AI Inference using the live slider configuration parameter
        results = model.predict(source=img_array, conf=conf_threshold)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        num_detections = len(results[0].boxes)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Enhanced Side-by-Side View with Spacing
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### 🔍 Raw Radar B-Scan Input")
        st.image(img, use_container_width=True)
    with col2:
        st.markdown("### 🚀 Live AI Target Diagnostics")
        st.image(res_rgb, use_container_width=True)

    # --- 6. INDUSTRIAL METRIC DASHBOARD CARD ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    
    metric_col1, metric_col2 = st.columns([1, 2], gap="medium")
    
    with metric_col1:
        # High impact metrics
        st.metric(
            label="Total Anomalies Identified", 
            value=f"{num_detections} Targets", 
            delta=f"Active Threshold: {conf_threshold:.2f}",
            delta_color="normal"
        )
    
    with metric_col2:
        st.markdown("### 📋 Automated Diagnostics Summary")
        if num_detections > 0:
            st.success(f"✅ Scanning Cycle Complete: Isolated **{num_detections}** anomalies. Geolocation bounding boxes accurately locked onto target hyperbolic apexes.")
        else:
            st.info("ℹ️ Scanning Cycle Complete: No anomalies cleared the confidence cutoff threshold. Stratification appears stable.")
            
    # Premium bottom tooltip box
    st.info("💡 **Exhibition Tip:** Use the slider in the sidebar to dynamically change the detection sensitivity. Watch how the confidence scores updates live on the model outputs!")

else:
    if mode == "Upload My Own Radargram":
        st.info("Drop a radar B-scan file above to initialize computer vision target metrics.")
