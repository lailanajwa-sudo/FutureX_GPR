import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. PREMIUM WINDOW CONFIGURATION ---
st.set_page_config(
    page_title="GPR-X Detection Suite", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. OFFICIAL ENTERPRISE DESIGN SYSTEM (CSS) ---
st.markdown("""
    <style>
    /* Hide default Streamlit clutter */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Global Font & Canvas Overrides */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0B0F19 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Brand Navigation Bar */
    .brand-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 0;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 40px;
    }
    .brand-logo {
        color: #FFFFFF;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .brand-status {
        color: #10B981;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        background: rgba(16, 185, 129, 0.1);
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    /* Data Presentation Grid headers */
    .section-title {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 15px;
    }
    
    /* Custom Data Metrics Card */
    .telemetry-card {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 24px;
        margin-top: 30px;
    }
    .telemetry-val {
        font-size: 2.8rem;
        font-weight: 700;
        color: #38BDF8;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        line-height: 1;
    }
    .telemetry-lbl {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 8px;
    }
    .telemetry-status {
        color: #E2E8F0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. BRAND HEADER ---
st.markdown("""
    <div class="brand-navbar">
        <div class="brand-logo">GPR-X // Advanced Analytics</div>
        <div class="brand-status">● CORE SYSTEM ONLINE</div>
    </div>
""", unsafe_allow_html=True)

# --- 4. CORE MODEL COUPLING ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Engine Core Error: 'best.pt' missing from server root.")

# --- 5. CONTROL ROUTING MATRIX ---
mode = st.radio(
    "INPUT CONTROL CHANNEL",
    ["PRE-LOADED INTERACTIVE PROFILES", "EXTERNAL IMAGE UPLOAD FRAME"],
    label_visibility="collapsed",
    horizontal=True
)

img = None

if mode == "EXTERNAL IMAGE UPLOAD FRAME":
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
else:
    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
    sample_options = {
        "DATASET PROFILE 01 // Lab Test — Buried Cavity": "samples/exp_cavity.png",
        "DATASET PROFILE 02 // Lab Test — Metal Pipe": "samples/exp_metal_pipe.png",
        "DATASET PROFILE 03 // Lab Test — Concrete Block": "samples/exp_concrete.png",
        "DATASET PROFILE 04 // Lab Test — Pipe & Cavity Combo": "samples/exp_pipe_cavity.png",
        "DATASET PROFILE 05 // Lab Test — Full Structural Mix": "samples/exp_all.png",
        "DATASET PROFILE 06 // Field Scan — Manhole Cover": "samples/real_manhole.jpg",
        "DATASET PROFILE 07 // Field Scan — Real-World Cavity": "samples/real_cavity.JPG"
    }
    selected_sample = st.selectbox("", list(sample_options.keys()), label_visibility="collapsed")
    sample_path = sample_options[selected_sample]
    
    if os.path.exists(sample_path):
        img = Image.open(sample_path).convert("RGB")
    else:
        st.markdown(f"<p style='color:#EF4444;'>IO Error: Dataset array missing at target path.</p>", unsafe_allow_html=True)

# --- 6. PROCESSING & DUAL-WINDOW ENGINE ---
if img is not None:
    img_array = np.array(img)

    with st.spinner(""):
        results = model.predict(source=img_array, conf=0.25, verbose=False)
        res_plotted = results[0].plot(labels=True, boxes=True)
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        num_detections = len(results[0].boxes)

    st.markdown("<div style='margin-top:35px;'></div>", unsafe_allow_html=True)
    
    # Grid Separation Layout
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="section-title">01 / SOURCE DATASTREAM INPUT</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">02 / COMPUTER VISION RADAR DIAGNOSTICS</div>', unsafe_allow_html=True)
        st.image(res_rgb, use_container_width=True)

    # --- 7. TECHNICAL TELEMETRY OVERVIEW ---
    status_text = (
        f"Neural network evaluation finalized. **{num_detections} anomalies successfully flagged** within the current frame layer matrix."
        if num_detections > 0 else 
        "Neural network evaluation finalized. Frame scan metrics read stable. Zero structural anomalies exceeded baseline verification criteria."
    )
    
    st.markdown(f"""
        <div class="telemetry-card">
            <div class="section-title" style="margin-bottom:15px;">03 / SYSTEM TELEMETRY SUMMARY</div>
            <table style="width:100%; border-collapse:collapse;">
                <tr>
                    <td style="width:25%; vertical-align:top; border-right:1px solid #1E293B; padding-right:20px;">
                        <div class="telemetry-val">{num_detections:02d}</div>
                        <div class="telemetry-lbl">REGISTRATION EVENTS</div>
                    </td>
                    <td style="width:75%; vertical-align:middle; padding-left:35px;">
                        <div class="telemetry-status">{status_text}</div>
                    </td>
                </tr>
            </table>
        </div>
    """, unsafe_allow_html=True)
            
else:
    if mode == "EXTERNAL IMAGE UPLOAD FRAME":
        st.markdown("<br><p style='color:#475569; font-size:0.9rem; font-family:monospace;'>SYSTEM IDLE // AWAITING B-SCAN ATTACHMENT STREAM...</p>", unsafe_allow_html=True)
