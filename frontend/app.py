import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
import base64
from io import BytesIO
import streamlit.components.v1 as components

# Configure page
st.set_page_config(page_title="AgriAI - Smart Farming", page_icon="🌾", layout="wide")

# ═══════════════════════════════════════════════════════════════
# Helper: Load 3D HTML page
# ═══════════════════════════════════════════════════════════════
def load_3d_page(filename, height=500):
    """Load a Three.js 3D animated HTML page from the 3d/ directory."""
    page_path = os.path.join(os.path.dirname(__file__), "3d", filename)
    if os.path.exists(page_path):
        with open(page_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=height, scrolling=False)
    else:
        st.warning(f"3D visualization not found: {filename}")

# ═══════════════════════════════════════════════════════════════
# 🎨 AGRICULTURAL FIELD THEME — Earthy, Warm, Farm-Fresh
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* ══ FIELD BACKGROUND ══ */
    .stApp {
        background: linear-gradient(160deg, 
            #1a2e1a 0%, 
            #2d4a2d 25%, 
            #1e3a1e 50%, 
            #2a3d1e 75%, 
            #1a2e1a 100%);
        color: #f0ead6;
    }

    /* Subtle grain texture overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 0;
    }

    /* ══ EARTHY CARD CONTAINERS ══ */
    .farm-card {
        background: linear-gradient(135deg, rgba(45, 74, 45, 0.6), rgba(42, 61, 30, 0.4));
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(168, 195, 128, 0.2);
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(168, 195, 128, 0.1);
        margin-bottom: 1.5rem;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .farm-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #c8a951, #7cb342, #c8a951);
        border-radius: 16px 16px 0 0;
    }
    .farm-card:hover {
        transform: translateY(-4px);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.35),
            inset 0 1px 0 rgba(168, 195, 128, 0.15);
        border-color: rgba(200, 169, 81, 0.35);
    }

    /* ══ GOLDEN WHEAT HEADERS ══ */
    h1 {
        color: #daa520 !important;
        font-weight: 900 !important;
        font-size: 2.4rem !important;
        text-shadow: 0 2px 8px rgba(218, 165, 32, 0.3);
        letter-spacing: -0.5px;
    }
    h2 {
        color: #c8a951 !important;
        font-weight: 700 !important;
        text-shadow: 0 1px 4px rgba(200, 169, 81, 0.2);
    }
    h3 {
        color: #a8c380 !important;
        font-weight: 600 !important;
    }
    h4 {
        color: #8fbc5a !important;
        font-weight: 600 !important;
    }

    /* ══ EARTHY GREEN BUTTONS ══ */
    .stButton>button {
        background: linear-gradient(135deg, #558b2f 0%, #33691e 50%, #2e7d32 100%);
        color: #f0ead6;
        border: 1px solid rgba(168, 195, 128, 0.3);
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(85, 139, 47, 0.35);
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #689f38 0%, #558b2f 50%, #33691e 100%);
        box-shadow: 0 6px 25px rgba(85, 139, 47, 0.5);
        transform: translateY(-2px);
        border-color: rgba(200, 169, 81, 0.5);
    }

    /* ══ FORM INPUTS ══ */
    .stNumberInput input, .stTextInput input, .stTextArea textarea {
        background-color: rgba(26, 46, 26, 0.7) !important;
        color: #f0ead6 !important;
        border: 1px solid rgba(168, 195, 128, 0.25) !important;
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: rgba(200, 169, 81, 0.6) !important;
        box-shadow: 0 0 0 2px rgba(200, 169, 81, 0.15) !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(26, 46, 26, 0.7) !important;
        border: 1px solid rgba(168, 195, 128, 0.25) !important;
        border-radius: 10px !important;
    }
    .stSelectbox div[data-baseweb="select"] * {
        color: #f0ead6 !important;
    }

    /* ══ SIDEBAR ══ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2e1a 0%, #243524 50%, #1a2e1a 100%) !important;
        border-right: 1px solid rgba(168, 195, 128, 0.15);
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #d4c897 !important;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #daa520 !important;
    }

    /* ══ METRIC DISPLAY ══ */
    .metric-value {
        font-size: 3.2rem;
        font-weight: 900;
        color: #daa520;
        text-shadow: 0 0 15px rgba(218, 165, 32, 0.4);
        text-align: center;
        font-family: 'Outfit', sans-serif;
    }
    .metric-label {
        font-size: 1.1rem;
        color: #a8c380;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-weight: 600;
    }

    /* ══ RESULT PANELS ══ */
    .result-success {
        background: linear-gradient(135deg, rgba(85, 139, 47, 0.15), rgba(46, 125, 50, 0.1));
        border: 2px solid rgba(124, 179, 66, 0.4);
        border-radius: 14px;
        padding: 2rem;
        margin-top: 1rem;
    }
    .result-info {
        background: linear-gradient(135deg, rgba(200, 169, 81, 0.12), rgba(218, 165, 32, 0.08));
        border: 2px solid rgba(200, 169, 81, 0.35);
        border-radius: 14px;
        padding: 2rem;
        margin-top: 1rem;
    }
    .result-warning {
        background: linear-gradient(135deg, rgba(183, 28, 28, 0.12), rgba(198, 40, 40, 0.08));
        border: 2px solid rgba(239, 83, 80, 0.35);
        border-radius: 14px;
        padding: 2rem;
        margin-top: 1rem;
    }

    /* ══ FEATURE CARDS (Home Page) ══ */
    .feature-icon {
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));
    }
    .feature-title {
        color: #c8a951;
        font-size: 1.15rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        color: #b8c9a0;
        font-size: 0.9rem;
        text-align: center;
        line-height: 1.5;
    }
    
    /* ══ CHAT BUBBLES ══ */
    .chat-user {
        background: linear-gradient(135deg, rgba(200, 169, 81, 0.2), rgba(218, 165, 32, 0.1));
        border: 1px solid rgba(200, 169, 81, 0.3);
        border-radius: 14px 14px 4px 14px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0 0.8rem 3rem;
        color: #f0ead6;
    }
    .chat-bot {
        background: linear-gradient(135deg, rgba(85, 139, 47, 0.15), rgba(46, 125, 50, 0.1));
        border: 1px solid rgba(124, 179, 66, 0.25);
        border-radius: 14px 14px 14px 4px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 3rem 0.8rem 0;
        color: #f0ead6;
    }

    /* ══ FILE UPLOADER ══ */
    .stFileUploader {
        border: 2px dashed rgba(168, 195, 128, 0.3) !important;
        border-radius: 14px !important;
        background: rgba(26, 46, 26, 0.4) !important;
    }

    /* ══ TABS ══ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(45, 74, 45, 0.4);
        border-radius: 10px 10px 0 0;
        color: #a8c380;
        border: 1px solid rgba(168, 195, 128, 0.15);
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(85, 139, 47, 0.3) !important;
        color: #daa520 !important;
        border-color: rgba(200, 169, 81, 0.4) !important;
    }

    /* ══ DIVIDERS ══ */
    hr {
        border-color: rgba(168, 195, 128, 0.15) !important;
    }

    /* ══ INFO/WARNING BOXES ══ */
    .stAlert {
        background: rgba(45, 74, 45, 0.4) !important;
        border: 1px solid rgba(168, 195, 128, 0.2) !important;
        border-radius: 12px !important;
        color: #f0ead6 !important;
    }

    /* ══ SPINNER ══ */
    .stSpinner > div > div {
        border-top-color: #daa520 !important;
    }

    /* ══ SCROLLBAR ══ */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #1a2e1a; }
    ::-webkit-scrollbar-thumb { background: rgba(168, 195, 128, 0.3); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(200, 169, 81, 0.5); }

    /* ══ 3D SECTION HEADER ══ */
    .viz-header {
        text-align: center;
        margin: 1rem 0;
        padding: 0.6rem 1.2rem;
        background: linear-gradient(90deg, transparent, rgba(218, 165, 32, 0.08), transparent);
        border-top: 1px solid rgba(218, 165, 32, 0.15);
        border-bottom: 1px solid rgba(218, 165, 32, 0.15);
    }
    .viz-header span {
        color: #c8a951;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 4px;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# API Configuration
# ═══════════════════════════════════════════════════════════════
API_URL = "http://127.0.0.1:8000"

# ═══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div style="font-size: 3rem; margin-bottom: 0.3rem;">🌾</div>
    <div style="color: #daa520; font-size: 1.5rem; font-weight: 800; letter-spacing: 1px;">AgriAI</div>
    <div style="color: #8fbc5a; font-size: 0.75rem; font-weight: 400; letter-spacing: 3px; text-transform: uppercase;">Smart Farming Platform</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate", 
    [
        "🏡 Home",
        "📈 Yield Predictor",
        "💊 Fertilizer AI",
        "🍃 Disease Detector",
        "🤖 AgriBot",
        "📄 Soil Report Reader",
        "🛰️ Field Analysis",
        "📊 Crop Analytics",
        "🔬 Deep Insights"
    ], 
    label_visibility="hidden"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 0.8rem; background: rgba(45, 74, 45, 0.4); border-radius: 10px; border: 1px solid rgba(168, 195, 128, 0.15);">
    <div style="color: #c8a951; font-weight: 600; font-size: 0.85rem; margin-bottom: 0.4rem;">🌍 Coverage Area</div>
    <div style="color: #b8c9a0; font-size: 0.8rem; line-height: 1.6;">
        Punjab • Haryana<br>
        Uttar Pradesh • Rajasthan
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="padding: 0.8rem; margin-top: 0.8rem; background: rgba(45, 74, 45, 0.4); border-radius: 10px; border: 1px solid rgba(168, 195, 128, 0.15);">
    <div style="color: #c8a951; font-weight: 600; font-size: 0.85rem; margin-bottom: 0.4rem;">🌱 Supported Crops</div>
    <div style="color: #b8c9a0; font-size: 0.8rem; line-height: 1.6;">
        Wheat • Rice<br>
        Sugarcane • Mustard
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="padding: 0.8rem; margin-top: 0.8rem; background: rgba(45, 74, 45, 0.4); border-radius: 10px; border: 1px solid rgba(124,179,66,0.15);">
    <div style="color: #7cb342; font-weight: 600; font-size: 0.85rem; margin-bottom: 0.4rem;">🎮 3D Visualizations</div>
    <div style="color: #b8c9a0; font-size: 0.8rem; line-height: 1.6;">
        Each model features an<br>
        interactive Three.js 3D<br>
        animated visualization
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 🏡 HOME PAGE
# ═══════════════════════════════════════════════════════════════
if page == "🏡 Home":
    st.markdown("<h1>🌾 AgriAI — Precision Farming Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="farm-card">
        <h3 style="margin-top: 0;">Welcome to the Future of North Indian Agriculture</h3>
        <p style="color: #b8c9a0; font-size: 1.05rem; line-height: 1.8;">
            AgriAI combines <strong style="color: #c8a951;">Machine Learning</strong>, 
            <strong style="color: #c8a951;">Computer Vision</strong>, and 
            <strong style="color: #c8a951;">Natural Language Processing</strong> 
            to deliver actionable insights for farmers. From predicting crop yields to detecting 
            plant diseases from a simple photo — your smart farming companion is here.
            <br><br>
            <strong style="color: #7cb342;">✨ NEW:</strong> Every model page now features immersive 
            <strong style="color: #daa520;">Three.js 3D animated visualizations</strong> that bring 
            your agricultural data to life!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards Grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="farm-card" style="text-align: center; min-height: 220px;">
            <div class="feature-icon">🚜</div>
            <div class="feature-title">Yield Predictor</div>
            <div class="feature-desc">Predict harvest output using soil chemistry, nutrients, and live weather data from Open-Meteo API. <em style="color:#7cb342;">3D Bar Chart</em></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="farm-card" style="text-align: center; min-height: 220px;">
            <div class="feature-icon">🧪</div>
            <div class="feature-title">Fertilizer AI</div>
            <div class="feature-desc">Get precise NPK fertilizer recommendations based on your soil's exact nutrient profile. <em style="color:#7cb342;">3D Molecule</em></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="farm-card" style="text-align: center; min-height: 220px;">
            <div class="feature-icon">🍃</div>
            <div class="feature-title">Disease Detector</div>
            <div class="feature-desc">Snap a photo of a diseased leaf — our AI identifies the disease and suggests treatment. <em style="color:#7cb342;">3D Leaf Scanner</em></div>
        </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("""
        <div class="farm-card" style="text-align: center; min-height: 220px;">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AgriBot</div>
            <div class="feature-desc">Chat with our AI farming expert. Ask anything about crops, pests, soil, or government schemes. <em style="color:#7cb342;">3D Robot</em></div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="farm-card" style="text-align: center; min-height: 220px;">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Soil Report Reader</div>
            <div class="feature-desc">Upload your soil test report PDF — we auto-extract NPK, pH values using OCR. <em style="color:#7cb342;">3D Document Stack</em></div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown("""
        <div class="farm-card" style="text-align: center; min-height: 220px;">
            <div class="feature-icon">🛰️</div>
            <div class="feature-title">Field Health</div>
            <div class="feature-desc">Analyze drone or satellite field images. See healthy vs. stressed crop zones as a heatmap. <em style="color:#7cb342;">3D Terrain</em></div>
        </div>
        """, unsafe_allow_html=True)

    # Tech Stack Banner
    st.markdown("---")
    st.markdown("""
    <div class="farm-card" style="text-align: center;">
        <h4 style="margin-top: 0;">⚡ Powered By</h4>
        <div style="display: flex; justify-content: center; gap: 2.5rem; flex-wrap: wrap; margin-top: 1rem;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">🧠</div>
                <div style="color: #c8a951; font-size: 0.85rem; font-weight: 600;">XGBoost</div>
                <div style="color: #8a9a6f; font-size: 0.7rem;">ML Models</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">👁️</div>
                <div style="color: #c8a951; font-size: 0.85rem; font-weight: 600;">Computer Vision</div>
                <div style="color: #8a9a6f; font-size: 0.7rem;">Disease Detection</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">💬</div>
                <div style="color: #c8a951; font-size: 0.85rem; font-weight: 600;">NLP Chatbot</div>
                <div style="color: #8a9a6f; font-size: 0.7rem;">Gemini AI</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">⚡</div>
                <div style="color: #c8a951; font-size: 0.85rem; font-weight: 600;">FastAPI</div>
                <div style="color: #8a9a6f; font-size: 0.7rem;">Backend</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">🎮</div>
                <div style="color: #c8a951; font-size: 0.85rem; font-weight: 600;">Three.js</div>
                <div style="color: #8a9a6f; font-size: 0.7rem;">3D Animations</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">🌤️</div>
                <div style="color: #c8a951; font-size: 0.85rem; font-weight: 600;">Open-Meteo</div>
                <div style="color: #8a9a6f; font-size: 0.7rem;">Weather API</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 📈 YIELD PREDICTOR
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Yield Predictor":
    st.markdown("<h1>📈 Crop Yield Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #b8c9a0; font-size: 1.05rem;'>Enter your soil parameters below. Weather fields auto-fetch from Open-Meteo if left blank.</p>", unsafe_allow_html=True)

    # 3D Visualization
    st.markdown('<div class="viz-header"><span>🎮 3D Yield Visualization</span></div>', unsafe_allow_html=True)
    load_3d_page("yield_3d.html", height=420)

    st.markdown('<div class="farm-card">', unsafe_allow_html=True)
    st.markdown("### 🌱 Crop & Location")
    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("State", ["Punjab", "Haryana", "UP", "Rajasthan"], key="yield_state")
    with col2:
        crop = st.selectbox("Crop Type", ["Wheat", "Rice", "Sugarcane", "Mustard"], key="yield_crop")

    st.markdown("### 🧪 Soil Chemistry")
    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50, key="yield_n")
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1, key="yield_ph")
    with col2:
        p = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=30, key="yield_p")
        moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=40.0, key="yield_moisture")
    with col3:
        k = st.number_input("Potassium (K)", min_value=0, max_value=200, value=30, key="yield_k")
        org_content = st.number_input("Organic Content (%)", min_value=0.0, max_value=10.0, value=1.5, key="yield_org")

    st.markdown("### ☁️ Weather <span style='color: #8a9a6f; font-size: 0.85rem;'>(Leave blank for auto-fetch)</span>", unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3)
    with w1:
        temp = st.number_input("Temperature (°C)", value=None, placeholder="Auto-fetch", key="yield_temp")
    with w2:
        hum = st.number_input("Humidity (%)", value=None, placeholder="Auto-fetch", key="yield_hum")
    with w3:
        rain = st.number_input("Rainfall (mm)", value=None, placeholder="Auto-fetch", key="yield_rain")

    if st.button("🌾 Predict Yield", key="yield_btn"):
        payload = {
            "State": state, "Crop": crop, "N": n, "P": p, "K": k,
            "pH": ph, "Moisture": moisture, "Organic_Content": org_content,
            "Temperature": temp, "Humidity": hum, "Rainfall": rain
        }
        try:
            with st.spinner("🌤️ Fetching weather & analyzing soil data..."):
                res = requests.post(f"{API_URL}/predict/yield", json=payload)
                res.raise_for_status()
                data = res.json()

                st.markdown(f"""
                <div class="result-success">
                    <div class="metric-label">Estimated Yield</div>
                    <div class="metric-value">{data['predicted_yield_tons_ha']} {data['unit']}</div>
                    <p style="text-align: center; color: #a8c380; margin-top: 12px; font-size: 0.95rem;">
                        {'☁️ Weather data auto-fetched via Open-Meteo API' if data['weather_auto_fetched'] else '☀️ Manual weather values used'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Could not connect to backend: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 💊 FERTILIZER AI
# ═══════════════════════════════════════════════════════════════
elif page == "💊 Fertilizer AI":
    st.markdown("<h1>💊 Intelligent Fertilizer Prescription</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #b8c9a0; font-size: 1.05rem;'>Get the perfect nutrient mix recommendation for your soil conditions.</p>", unsafe_allow_html=True)

    # 3D Visualization
    st.markdown('<div class="viz-header"><span>🎮 3D NPK Molecule</span></div>', unsafe_allow_html=True)
    load_3d_page("fertilizer_3d.html", height=420)

    st.markdown('<div class="farm-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌱 Crop & Location")
        state = st.selectbox("State", ["Punjab", "Haryana", "UP", "Rajasthan"], key="fert_state")
        crop = st.selectbox("Crop Type", ["Wheat", "Rice", "Sugarcane", "Mustard"], key="fert_crop")
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1, key="fert_ph")
        moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=40.0, key="fert_moisture")
    with col2:
        st.markdown("### 🧪 Soil Nutrients")
        n = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=40, key="fert_n")
        p = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=20, key="fert_p")
        k = st.number_input("Potassium (K)", min_value=0, max_value=200, value=20, key="fert_k")
        org_content = st.number_input("Organic Content (%)", min_value=0.0, max_value=10.0, value=1.0, key="fert_org")

    if st.button("🔬 Get Recommendation", key="fert_btn"):
        payload = {
            "State": state, "Crop": crop, "N": n, "P": p, "K": k,
            "pH": ph, "Moisture": moisture, "Organic_Content": org_content
        }
        try:
            with st.spinner("🧪 Analyzing soil nutrient profile..."):
                res = requests.post(f"{API_URL}/predict/fertilizer", json=payload)
                res.raise_for_status()
                data = res.json()

                st.markdown(f"""
                <div class="result-info">
                    <div class="metric-label">Recommended Fertilizer</div>
                    <div class="metric-value" style="font-size: 2.4rem;">{data['recommended_fertilizer']}</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Could not connect to backend: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 🍃 DISEASE DETECTOR (CV)
# ═══════════════════════════════════════════════════════════════
elif page == "🍃 Disease Detector":
    st.markdown("<h1>🍃 Crop Disease Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #b8c9a0; font-size: 1.05rem;'>Upload a photo of a diseased crop leaf. Our AI will identify the disease and recommend treatment.</p>", unsafe_allow_html=True)

    # 3D Visualization
    st.markdown('<div class="viz-header"><span>🎮 3D Leaf Health Scanner</span></div>', unsafe_allow_html=True)
    load_3d_page("disease_3d.html", height=420)

    st.markdown('<div class="farm-card">', unsafe_allow_html=True)
    
    upload_tab, camera_tab = st.tabs(["📁 Upload Image", "📸 Take Photo"])
    
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload a leaf image (JPG, PNG)", 
            type=["jpg", "jpeg", "png"], 
            key="disease_upload",
            help="Take a clear close-up photo of the affected leaf"
        )
    
    with camera_tab:
        camera_file = st.camera_input("Take a photo of the affected leaf", key="disease_camera")
    
    image_file = uploaded_file or camera_file
    
    if image_file:
        col_img, col_result = st.columns([1, 1])
        with col_img:
            st.markdown("#### 📷 Uploaded Image")
            st.image(image_file, use_container_width=True)
        
        with col_result:
            if st.button("🔍 Analyze Disease", key="disease_btn"):
                try:
                    with st.spinner("🧬 AI is analyzing the leaf..."):
                        files = {"file": (image_file.name if hasattr(image_file, 'name') else "photo.jpg", image_file.getvalue(), "image/jpeg")}
                        res = requests.post(f"{API_URL}/predict/disease", files=files)
                        res.raise_for_status()
                        data = res.json()

                        if data.get("is_healthy", False):
                            st.markdown(f"""
                            <div class="result-success">
                                <div class="feature-icon">✅</div>
                                <div class="metric-label">Plant Health Status</div>
                                <div class="metric-value" style="color: #7cb342; font-size: 2.2rem;">Healthy</div>
                                <p style="text-align: center; color: #a8c380; margin-top: 8px;">
                                    Confidence: {data.get('confidence', 0)*100:.1f}%
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-warning">
                                <div class="feature-icon">⚠️</div>
                                <div class="metric-label">Disease Detected</div>
                                <div class="metric-value" style="color: #ef5350; font-size: 2rem;">{data.get('disease_name', 'Unknown')}</div>
                                <p style="text-align: center; color: #e0a0a0; margin-top: 8px;">
                                    Confidence: {data.get('confidence', 0)*100:.1f}%
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if data.get("treatment"):
                                st.markdown(f"""
                                <div class="result-info" style="margin-top: 1rem;">
                                    <h4 style="margin-top: 0;">💊 Recommended Treatment</h4>
                                    <p style="color: #d4c897; line-height: 1.7;">{data['treatment']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"⚠️ Disease detection failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 🤖 AGRIBOT CHATBOT (NLP)
# ═══════════════════════════════════════════════════════════════
elif page == "🤖 AgriBot":
    st.markdown("<h1>🤖 AgriBot — Your AI Farming Expert</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #b8c9a0; font-size: 1.05rem;'>Ask any question about farming, crops, pests, fertilizers, or government schemes.</p>", unsafe_allow_html=True)

    # 3D Visualization
    st.markdown('<div class="viz-header"><span>🎮 3D AgriBot Assistant</span></div>', unsafe_allow_html=True)
    load_3d_page("agribot_3d.html", height=380)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat display
    st.markdown('<div class="farm-card" style="min-height: 300px; max-height: 500px; overflow-y: auto;">', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; color: #8a9a6f;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌱</div>
            <div style="font-size: 1.1rem; font-weight: 500;">Start a conversation!</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">Try asking: "My wheat is turning yellow, what should I do?"</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-user">
                    <strong style="color: #c8a951;">You:</strong><br>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-bot">
                    <strong style="color: #7cb342;">🌾 AgriBot:</strong><br>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_message = st.text_input(
            "Type your farming question...", 
            key="chat_input", 
            label_visibility="collapsed",
            placeholder="Ask me anything about farming..."
        )
    with col_send:
        send_clicked = st.button("🌾 Send", key="chat_send")

    if send_clicked and user_message:
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        try:
            with st.spinner("🌱 AgriBot is thinking..."):
                payload = {
                    "message": user_message,
                    "history": st.session_state.chat_history[:-1]
                }
                res = requests.post(f"{API_URL}/chat", json=payload)
                res.raise_for_status()
                data = res.json()
                bot_response = data.get("response", "Sorry, I couldn't process your question. Please try again.")
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                st.rerun()
        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Could not connect to AgriBot service. Error: {str(e)}"})
            st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# 📄 SOIL REPORT READER (NLP + OCR)
# ═══════════════════════════════════════════════════════════════
elif page == "📄 Soil Report Reader":
    st.markdown("<h1>📄 Soil Report Reader</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #b8c9a0; font-size: 1.05rem;'>Upload your soil test report (PDF or image). Our OCR + NLP engine will extract NPK, pH values automatically.</p>", unsafe_allow_html=True)

    # 3D Visualization
    st.markdown('<div class="viz-header"><span>🎮 3D Document Scanner</span></div>', unsafe_allow_html=True)
    load_3d_page("report_3d.html", height=420)

    st.markdown('<div class="farm-card">', unsafe_allow_html=True)
    uploaded_report = st.file_uploader(
        "Upload Soil Test Report", 
        type=["pdf", "jpg", "jpeg", "png"],
        key="report_upload",
        help="Upload your Soil Health Card or lab report (PDF, JPG, PNG)"
    )
    
    if uploaded_report:
        st.markdown(f"""
        <div style="background: rgba(45, 74, 45, 0.5); padding: 0.8rem 1.2rem; border-radius: 10px; color: #a8c380; margin: 1rem 0;">
            📎 Uploaded: <strong>{uploaded_report.name}</strong> ({uploaded_report.size / 1024:.1f} KB)
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📖 Extract Values", key="report_btn"):
            try:
                with st.spinner("🔍 Reading and parsing your soil report..."):
                    files = {"file": (uploaded_report.name, uploaded_report.getvalue(), uploaded_report.type)}
                    res = requests.post(f"{API_URL}/parse/soil-report", files=files)
                    res.raise_for_status()
                    data = res.json()
                    
                    extracted = data.get("extracted_values", {})
                    
                    st.markdown("### 📋 Extracted Values")
                    col1, col2, col3 = st.columns(3)
                    
                    params = [
                        ("Nitrogen (N)", extracted.get("N"), "kg/ha"),
                        ("Phosphorus (P)", extracted.get("P"), "kg/ha"),
                        ("Potassium (K)", extracted.get("K"), "kg/ha"),
                        ("Soil pH", extracted.get("pH"), ""),
                        ("Moisture", extracted.get("Moisture"), "%"),
                        ("Organic Content", extracted.get("Organic_Content"), "%")
                    ]
                    
                    for i, (name, value, unit) in enumerate(params):
                        with [col1, col2, col3][i % 3]:
                            display_val = f"{value} {unit}" if value is not None else "—"
                            color = "#7cb342" if value is not None else "#ef5350"
                            st.markdown(f"""
                            <div style="background: rgba(45, 74, 45, 0.5); padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 0.8rem; border: 1px solid rgba(168, 195, 128, 0.15);">
                                <div style="color: #8a9a6f; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">{name}</div>
                                <div style="color: {color}; font-size: 1.6rem; font-weight: 800; margin: 0.3rem 0;">{display_val}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if data.get("parsing_notes"):
                        st.info(f"📝 Notes: {data['parsing_notes']}")
                    
                    # Offer to use values in Yield Predictor
                    st.markdown("""
                    <div class="result-info" style="margin-top: 1rem;">
                        <p style="color: #d4c897; text-align: center;">
                            💡 <strong>Tip:</strong> Use these extracted values in the Yield Predictor or Fertilizer AI pages for accurate predictions!
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Could not parse the report: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 🛰️ FIELD ANALYSIS (CV)
# ═══════════════════════════════════════════════════════════════
elif page == "🛰️ Field Analysis":
    st.markdown("<h1>🛰️ Field Health Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #b8c9a0; font-size: 1.05rem;'>Upload a drone or satellite image of your field. Our AI generates a vegetation health heatmap.</p>", unsafe_allow_html=True)

    # 3D Visualization
    st.markdown('<div class="viz-header"><span>🎮 3D Terrain Health Map</span></div>', unsafe_allow_html=True)
    load_3d_page("field_3d.html", height=420)

    analysis_tab1, analysis_tab2 = st.tabs(["🛰️ Field Drone Analysis", "🌱 Soil Image Analysis"])
    
    with analysis_tab1:
        st.markdown('<div class="farm-card">', unsafe_allow_html=True)
        field_image = st.file_uploader(
            "Upload field/drone image",
            type=["jpg", "jpeg", "png"],
            key="field_upload",
            help="Upload an aerial or satellite view of your agricultural field"
        )
        
        if field_image:
            col_orig, col_heat = st.columns(2)
            with col_orig:
                st.markdown("#### 📷 Original Image")
                st.image(field_image, use_container_width=True)
            
            if st.button("🔍 Analyze Field Health", key="field_btn"):
                try:
                    with st.spinner("🛰️ Analyzing vegetation patterns..."):
                        files = {"file": (field_image.name, field_image.getvalue(), "image/jpeg")}
                        res = requests.post(f"{API_URL}/analyze/field", files=files)
                        res.raise_for_status()
                        data = res.json()
                        
                        with col_heat:
                            st.markdown("#### 🗺️ Health Heatmap")
                            if data.get("heatmap_base64"):
                                heatmap_bytes = base64.b64decode(data["heatmap_base64"])
                                st.image(heatmap_bytes, use_container_width=True)
                        
                        # Statistics
                        st.markdown("### 📊 Field Health Report")
                        m1, m2, m3 = st.columns(3)
                        health_pct = data.get("health_percentage", 0)
                        stress_pct = data.get("stress_percentage", 0)
                        moderate_pct = 100 - health_pct - stress_pct
                        
                        with m1:
                            st.markdown(f"""
                            <div class="result-success" style="text-align: center;">
                                <div class="metric-label" style="color: #7cb342;">Healthy</div>
                                <div class="metric-value" style="color: #7cb342; font-size: 2.5rem;">{health_pct:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m2:
                            st.markdown(f"""
                            <div class="result-info" style="text-align: center;">
                                <div class="metric-label" style="color: #c8a951;">Moderate</div>
                                <div class="metric-value" style="color: #c8a951; font-size: 2.5rem;">{moderate_pct:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m3:
                            st.markdown(f"""
                            <div class="result-warning" style="text-align: center;">
                                <div class="metric-label" style="color: #ef5350;">Stressed</div>
                                <div class="metric-value" style="color: #ef5350; font-size: 2.5rem;">{stress_pct:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"⚠️ Field analysis failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with analysis_tab2:
        st.markdown('<div class="farm-card">', unsafe_allow_html=True)

        # 3D Soil visualization
        st.markdown('<div class="viz-header"><span>🎮 3D Soil Block Analyzer</span></div>', unsafe_allow_html=True)
        load_3d_page("soil_3d.html", height=380)

        soil_image = st.file_uploader(
            "Upload soil image",
            type=["jpg", "jpeg", "png"],
            key="soil_upload",
            help="Photograph your soil from close up for color analysis"
        )
        
        if soil_image:
            col_img, col_res = st.columns([1, 1])
            with col_img:
                st.markdown("#### 📷 Soil Sample")
                st.image(soil_image, use_container_width=True)
            
            if st.button("🔬 Analyze Soil", key="soil_img_btn"):
                try:
                    with st.spinner("🌱 Analyzing soil color profile..."):
                        files = {"file": (soil_image.name, soil_image.getvalue(), "image/jpeg")}
                        res = requests.post(f"{API_URL}/analyze/soil-image", files=files)
                        res.raise_for_status()
                        data = res.json()
                        
                        with col_res:
                            quality = data.get("quality_label", "Unknown")
                            score = data.get("quality_score", 0)
                            
                            color_map = {
                                "Excellent": "#7cb342",
                                "Good": "#8bc34a", 
                                "Fair": "#c8a951",
                                "Poor": "#ef5350"
                            }
                            q_color = color_map.get(quality, "#c8a951")
                            
                            st.markdown(f"""
                            <div class="result-success" style="text-align: center;">
                                <div class="metric-label">Soil Quality</div>
                                <div class="metric-value" style="color: {q_color}; font-size: 2.5rem;">{quality}</div>
                                <div style="color: #a8c380; font-size: 1.1rem; margin-top: 0.5rem;">Score: {score}/100</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if data.get("recommendations"):
                            st.markdown("### 💡 Recommendations")
                            for rec in data["recommendations"]:
                                st.markdown(f"""
                                <div style="background: rgba(45, 74, 45, 0.4); padding: 0.7rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid #c8a951; color: #d4c897;">
                                    🌿 {rec}
                                </div>
                                """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"⚠️ Soil analysis failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 📊 CROP ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Crop Analytics":
    st.markdown("<h1>📊 Crop Production Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #b8c9a0; font-size: 1.05rem;'>Exploratory analysis of regional crop yields, soil conditions, and fertilizer recommendations.</p>", unsafe_allow_html=True)

    # Load dataset
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, 'backend', 'data', 'crop_data_proxy.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        try:
            import sys
            sys.path.append(os.path.join(base_dir, 'backend'))
            from data import generate_proxy_data
            df = generate_proxy_data(1000)
        except Exception:
            df = pd.DataFrame({
                "State": ["Punjab", "Haryana", "UP", "Rajasthan"] * 25,
                "Crop": ["Wheat", "Rice", "Sugarcane", "Mustard"] * 25,
                "Yield_tons_per_ha": np.random.uniform(1.0, 80.0, 100),
                "Fertilizer": ["Urea High Dose", "DAP", "NPK 19:19:19 + Lime", "Standard NPK 12:32:16"] * 25,
                "pH": np.random.uniform(5.5, 8.5, 100),
                "Moisture": np.random.uniform(20.0, 60.0, 100)
            })

    # Chart color scheme — earthy palette
    earth_colors = ["#7cb342", "#c8a951", "#8d6e63", "#5d8a3c", "#d4a017", "#a1887f"]

    st.markdown('<div class="farm-card">', unsafe_allow_html=True)
    st.markdown("### 🌾 State-wise Average Crop Yields")
    yield_avg = df.groupby(["State", "Crop"])["Yield_tons_per_ha"].mean().reset_index()
    fig1 = px.bar(
        yield_avg, x="State", y="Yield_tons_per_ha", color="Crop", barmode="group",
        labels={"Yield_tons_per_ha": "Average Yield (Tons/Ha)", "State": "State"},
        color_discrete_sequence=earth_colors
    )
    fig1.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)", 
        font_color="#d4c897",
        legend_title_font_color="#c8a951",
        xaxis=dict(gridcolor="rgba(168, 195, 128, 0.1)"),
        yaxis=dict(gridcolor="rgba(168, 195, 128, 0.1)")
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="farm-card">', unsafe_allow_html=True)
        st.markdown("### 🧪 Fertilizer Distribution")
        fert_counts = df["Fertilizer"].value_counts().reset_index()
        fert_counts.columns = ["Fertilizer", "Count"]
        fig2 = px.pie(
            fert_counts, names="Fertilizer", values="Count",
            color_discrete_sequence=earth_colors,
            hole=0.4
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#d4c897")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="farm-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Soil pH vs. Crop Yield")
        fig3 = px.scatter(
            df, x="pH", y="Yield_tons_per_ha", color="Crop",
            labels={"pH": "Soil pH", "Yield_tons_per_ha": "Yield (Tons/Ha)"},
            color_discrete_sequence=earth_colors
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font_color="#d4c897",
            legend_title_font_color="#c8a951",
            xaxis=dict(gridcolor="rgba(168, 195, 128, 0.1)"),
            yaxis=dict(gridcolor="rgba(168, 195, 128, 0.1)")
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 🔬 DEEP INSIGHTS
# ═══════════════════════════════════════════════════════════════
elif page == "🔬 Deep Insights":
    st.markdown("<h1>🔬 Advanced Agronomic Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #b8c9a0; font-size: 1.05rem;'>Deep statistical analysis, soil chemistry heatmaps, and NPK 3D clustering profiles.</p>", unsafe_allow_html=True)

    # Load dataset
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, 'backend', 'data', 'crop_data_proxy.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        try:
            import sys
            sys.path.append(os.path.join(base_dir, 'backend'))
            from data import generate_proxy_data
            df = generate_proxy_data(1000)
        except Exception:
            df = pd.DataFrame({
                "State": ["Punjab", "Haryana", "UP", "Rajasthan"] * 25,
                "Crop": ["Wheat", "Rice", "Sugarcane", "Mustard"] * 25,
                "Yield_tons_per_ha": np.random.uniform(1.0, 80.0, 100),
                "Fertilizer": ["Urea High Dose", "DAP", "NPK 19:19:19 + Lime", "Standard NPK 12:32:16"] * 25,
                "pH": np.random.uniform(5.5, 8.5, 100),
                "Moisture": np.random.uniform(20.0, 60.0, 100),
                "N": np.random.uniform(20, 120, 100),
                "P": np.random.uniform(10, 60, 100),
                "K": np.random.uniform(10, 60, 100)
            })

    earth_colors = ["#7cb342", "#c8a951", "#8d6e63", "#5d8a3c", "#d4a017", "#a1887f"]

    # Statistical Summary
    st.markdown('<div class="farm-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Descriptive Statistical Summary")
    stats_df = df.describe().T.rename(columns={"mean": "Average", "std": "Std Dev", "50%": "Median"})
    st.dataframe(stats_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="farm-card">', unsafe_allow_html=True)
        st.markdown("### 🌡️ Correlation Heatmap")
        num_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[num_cols].corr()
        fig_heat = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            color_continuous_scale=["#1a2e1a", "#558b2f", "#c8a951", "#daa520"]
        )
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#d4c897")
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="farm-card">', unsafe_allow_html=True)
        st.markdown("### 📦 Yield Distribution by Crop & State")
        fig_box = px.box(
            df, x="Crop", y="Yield_tons_per_ha", color="State",
            color_discrete_sequence=earth_colors
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font_color="#d4c897",
            legend_title_font_color="#c8a951",
            xaxis=dict(gridcolor="rgba(168, 195, 128, 0.1)"),
            yaxis=dict(gridcolor="rgba(168, 195, 128, 0.1)")
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3D Scatter Plot
    st.markdown('<div class="farm-card">', unsafe_allow_html=True)
    st.markdown("### 🔮 3D NPK Nutrient Space by Fertilizer")
    fig_3d = px.scatter_3d(
        df, x="N", y="P", z="K", color="Fertilizer",
        color_discrete_sequence=earth_colors
    )
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(168, 195, 128, 0.15)", color="#d4c897"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(168, 195, 128, 0.15)", color="#d4c897"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(168, 195, 128, 0.15)", color="#d4c897")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#d4c897"
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
