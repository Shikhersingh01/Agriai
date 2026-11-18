from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import joblib
import pandas as pd
import os
import sys
import logging

# Add backend dir to path for imports
backend_dir = os.path.dirname(__file__)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from data import fetch_weather_by_location

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AgriAI — Smart Farming API",
    description="ML, Computer Vision, and NLP endpoints for precision agriculture",
    version="2.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════

class CropYieldRequest(BaseModel):
    State: str
    Crop: str
    N: float
    P: float
    K: float
    pH: float
    Moisture: float
    Organic_Content: float
    Temperature: Optional[float] = None
    Humidity: Optional[float] = None
    Rainfall: Optional[float] = None

class FertilizerRequest(BaseModel):
    State: str
    Crop: str
    N: float
    P: float
    K: float
    pH: float
    Moisture: float
    Organic_Content: float

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict]] = None

# ═══════════════════════════════════════════════════════════
# ML Model Loading (Original XGBoost)
# ═══════════════════════════════════════════════════════════

yield_model = None
fert_model = None
fert_label_encoder = None

# Global NLP chatbot instance
_chatbot_instance = None

@app.on_event("startup")
def load_models():
    global yield_model, fert_model, fert_label_encoder
    models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    try:
        yield_model = joblib.load(os.path.join(models_dir, 'yield_model.pkl'))
        fert_model = joblib.load(os.path.join(models_dir, 'fert_model.pkl'))
        fert_label_encoder = joblib.load(os.path.join(models_dir, 'fert_label_encoder.pkl'))
        logging.info("ML models loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading models. Have you trained them? {e}")

# ═══════════════════════════════════════════════════════════
# ORIGINAL ENDPOINTS (Yield + Fertilizer + Health)
# ═══════════════════════════════════════════════════════════

@app.post("/predict/yield")
def predict_yield(req: CropYieldRequest):
    if yield_model is None:
        raise HTTPException(status_code=500, detail="Yield model not loaded")

    data = req.model_dump()
    weather_fetched = False
    
    # Auto-fetch missing weather
    if data['Temperature'] is None or data['Humidity'] is None or data['Rainfall'] is None:
        weather = fetch_weather_by_location(data['State'])
        weather_fetched = True
        if data['Temperature'] is None: data['Temperature'] = weather['Temperature']
        if data['Humidity'] is None: data['Humidity'] = weather['Humidity']
        if data['Rainfall'] is None: data['Rainfall'] = weather['Rainfall']
        
    df = pd.DataFrame([data])
    try:
        prediction = yield_model.predict(df)[0]
        return {
            "predicted_yield_tons_ha": round(float(prediction), 2),
            "weather_auto_fetched": weather_fetched,
            "used_features": data,
            "unit": "Tons/Ha"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/fertilizer")
def predict_fertilizer(req: FertilizerRequest):
    if fert_model is None:
        raise HTTPException(status_code=500, detail="Fertilizer model not loaded")

    df = pd.DataFrame([req.model_dump()])
    try:
        pred_idx = fert_model.predict(df)[0]
        fert_name = fert_label_encoder.inverse_transform([pred_idx])[0]
        return {
            "recommended_fertilizer": fert_name,
            "inputs": req.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "2.0",
        "modules": {
            "ml": yield_model is not None,
            "nlp": True,
            "cv": True
        }
    }

# ═══════════════════════════════════════════════════════════
# 🍃 DISEASE DETECTOR (Computer Vision)
# ═══════════════════════════════════════════════════════════

@app.post("/predict/disease")
async def predict_disease(file: UploadFile = File(...)):
    """Detect crop disease from uploaded leaf image."""
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        from cv.disease_detector import detect_disease
        result = detect_disease(image_bytes)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Disease detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Disease detection failed: {str(e)}")

# ═══════════════════════════════════════════════════════════
# 🌱 SOIL IMAGE ANALYZER (Computer Vision)
# ═══════════════════════════════════════════════════════════

@app.post("/analyze/soil-image")
async def analyze_soil_image(file: UploadFile = File(...)):
    """Analyze soil quality from uploaded soil photograph."""
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        from cv.soil_analyzer import analyze_soil_image as _analyze_soil
        result = _analyze_soil(image_bytes)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Soil analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Soil image analysis failed: {str(e)}")

# ═══════════════════════════════════════════════════════════
# 🛰️ FIELD HEALTH ANALYZER (Computer Vision)
# ═══════════════════════════════════════════════════════════

@app.post("/analyze/field")
async def analyze_field(file: UploadFile = File(...)):
    """Analyze field vegetation health from drone/satellite image."""
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        from cv.field_analyzer import analyze_field_image
        result = analyze_field_image(image_bytes)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Field analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Field analysis failed: {str(e)}")

# ═══════════════════════════════════════════════════════════
# 🤖 AGRIBOT CHATBOT (NLP)
# ═══════════════════════════════════════════════════════════

def _get_chatbot():
    """Get or create global chatbot instance (lazy singleton)."""
    global _chatbot_instance
    if _chatbot_instance is None:
        from nlp.chatbot import AgriChatbot
        _chatbot_instance = AgriChatbot()
        logging.info("AgriBot chatbot initialized.")
    return _chatbot_instance

@app.post("/chat")
async def chat(req: ChatRequest):
    """Chat with AgriBot farming assistant."""
    try:
        chatbot = _get_chatbot()
        history = req.history or []
        response = chatbot.chat(req.message, history)
        return {"response": response}
    except Exception as e:
        logging.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"AgriBot error: {str(e)}")

# ═══════════════════════════════════════════════════════════
# 📄 SOIL REPORT READER (NLP + OCR)
# ═══════════════════════════════════════════════════════════

@app.post("/parse/soil-report")
async def parse_soil_report(file: UploadFile = File(...)):
    """Extract soil values from uploaded PDF or image report."""
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        filename = file.filename or "report.pdf"
        
        from nlp.report_parser import parse_soil_report as _parse_report
        result = _parse_report(file_bytes, filename)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Report parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Report parsing failed: {str(e)}")
