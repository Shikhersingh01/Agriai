import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os
import random
import requests
import logging
import spaces
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor, XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Location & Weather Ingestion ---
LOCATION_COORDS = {
    "Punjab": {"lat": 31.1471, "lon": 75.3412},
    "Haryana": {"lat": 29.0588, "lon": 76.0856},
    "UP": {"lat": 26.8467, "lon": 80.9462},
    "Rajasthan": {"lat": 27.0238, "lon": 74.2179}
}

CROPS = ["Wheat", "Rice", "Sugarcane", "Mustard"]

def fetch_weather_by_location(state_name: str):
    if state_name not in LOCATION_COORDS:
        state_name = "UP"
    coords = LOCATION_COORDS[state_name]
    lat, lon = coords["lat"], coords["lon"]
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json().get("current", {})
        return {
            "Temperature": data.get("temperature_2m", 25.0),
            "Humidity": data.get("relative_humidity_2m", 60.0),
            "Rainfall": data.get("precipitation", 0.0)
        }
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return {"Temperature": 28.0, "Humidity": 55.0, "Rainfall": 5.0}

def generate_proxy_data(n_samples=3000):
    np.random.seed(42)
    random.seed(42)
    data = []
    for _ in range(n_samples):
        state = random.choice(list(LOCATION_COORDS.keys()))
        crop = random.choice(CROPS)
        ph = round(random.uniform(5.5, 8.5), 2)
        organic_content = round(random.uniform(0.5, 2.5), 2)
        moisture = round(random.uniform(20.0, 60.0), 1)

        if crop in ["Wheat", "Mustard"]:
            temp = round(random.uniform(10.0, 25.0), 1)
            humidity = round(random.uniform(40.0, 70.0), 1)
            rainfall = round(random.uniform(10.0, 50.0), 1) 
        else:
            temp = round(random.uniform(25.0, 38.0), 1)
            humidity = round(random.uniform(60.0, 90.0), 1)
            rainfall = round(random.uniform(100.0, 400.0), 1)
            
        n = random.randint(20, 120)
        p = random.randint(10, 60)
        k = random.randint(10, 60)

        base_yield = {"Wheat": 3.5, "Rice": 4.0, "Sugarcane": 70.0, "Mustard": 1.5}[crop]
        ph_mod = 1.0 - abs(7.0 - ph) * 0.1
        water_mod = 1.0 if (moisture + rainfall > 100) else 0.8
        
        yield_val = round(max(0.1, base_yield * ph_mod * water_mod * random.uniform(0.8, 1.2)), 2)

        if n < 40:
            fert = "Urea High Dose"
        elif n < 80 and p < 30:
            fert = "DAP (Di-ammonium Phosphate)"
        elif ph < 6.0:
            fert = "NPK 19:19:19 + Lime"
        else:
            fert = "Standard NPK 12:32:16"

        data.append({
            "State": state, "Crop": crop, "N": n, "P": p, "K": k,
            "pH": ph, "Moisture": moisture, "Organic_Content": organic_content,
            "Temperature": temp, "Humidity": humidity, "Rainfall": rainfall,
            "Yield_tons_per_ha": yield_val, "Fertilizer": fert
        })
    return pd.DataFrame(data)

# --- Model Pipeline ---
class CropPredictorModel:
    def __init__(self):
        self.yield_model = None
        self.fert_model = None
        self.fert_label_encoder = LabelEncoder()

    def build_yield_pipeline(self):
        numeric_features = ['N', 'P', 'K', 'pH', 'Moisture', 'Organic_Content', 'Temperature', 'Humidity', 'Rainfall']
        categorical_features = ['State', 'Crop']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        self.yield_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
        ])
        return self.yield_model

    def build_fert_pipeline(self):
        numeric_features = ['N', 'P', 'K', 'pH', 'Moisture', 'Organic_Content']
        categorical_features = ['State', 'Crop']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        self.fert_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(n_estimators=100, max_depth=4, eval_metric='mlogloss', random_state=42))
        ])
        return self.fert_model

def train_fresh_models():
    logger.info("Training fresh XGBoost models...")
    df = generate_proxy_data(3000)
    predictor = CropPredictorModel()
    
    X_yield = df.drop(columns=['Yield_tons_per_ha', 'Fertilizer'])
    y_yield = df['Yield_tons_per_ha']
    yield_model = predictor.build_yield_pipeline()
    yield_model.fit(X_yield, y_yield)
    
    X_fert = df.drop(columns=['Yield_tons_per_ha', 'Fertilizer', 'Temperature', 'Humidity', 'Rainfall'])
    y_fert = predictor.fert_label_encoder.fit_transform(df['Fertilizer'])
    fert_model = predictor.build_fert_pipeline()
    fert_model.fit(X_fert, y_fert)
    
    return yield_model, fert_model, predictor.fert_label_encoder

def get_models():
    logger.info("Initializing native XGBoost models on ZeroGPU container boot...")
    return train_fresh_models()

yield_model, fert_model, fert_encoder = get_models()

@spaces.GPU
def predict_crop_and_fertilizer(state, crop, n, p, k, ph, moisture, organic_content, temp, humidity, rainfall):
    weather_status = "☀️ Manual Weather Used"
    if temp is None or temp == 0 or humidity is None or humidity == 0 or rainfall is None or rainfall == 0:
        w = fetch_weather_by_location(state)
        weather_status = f"☁️ Auto-Fetched Weather: Temp {w['Temperature']}°C, Humidity {w['Humidity']}%, Rain {w['Rainfall']}mm"
        if temp is None or temp == 0: temp = w['Temperature']
        if humidity is None or humidity == 0: humidity = w['Humidity']
        if rainfall is None or rainfall == 0: rainfall = w['Rainfall']

    yield_input = {
        "State": state, "Crop": crop, "N": n, "P": p, "K": k,
        "pH": ph, "Moisture": moisture, "Organic_Content": organic_content,
        "Temperature": temp, "Humidity": humidity, "Rainfall": rainfall
    }
    df_yield = pd.DataFrame([yield_input])
    yield_val = round(float(yield_model.predict(df_yield)[0]), 2)

    fert_input = {
        "State": state, "Crop": crop, "N": n, "P": p, "K": k,
        "pH": ph, "Moisture": moisture, "Organic_Content": organic_content
    }
    df_fert = pd.DataFrame([fert_input])
    fert_idx = fert_model.predict(df_fert)[0]
    fert_name = fert_encoder.inverse_transform([fert_idx])[0]

    yield_output = f"Estimated Crop Yield: {yield_val} Tons/Ha\n({weather_status})"
    fert_output = f"Recommended Strategy: {fert_name}"

    return yield_output, fert_output

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🌾 AgriAI: North India Crop & Fertilizer Prediction System")
    gr.Markdown("XGBoost ML Models & Live Open-Meteo Weather Integration for Punjab, Haryana, UP & Rajasthan.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📍 Location & Soil Parameters")
            state = gr.Dropdown(["Punjab", "Haryana", "UP", "Rajasthan"], label="State", value="Punjab")
            crop = gr.Dropdown(["Wheat", "Rice", "Sugarcane", "Mustard"], label="Crop Type", value="Wheat")
            ph = gr.Slider(0.0, 14.0, value=6.5, step=0.1, label="Soil pH")
            moisture = gr.Slider(0.0, 100.0, value=40.0, label="Soil Moisture (%)")
            organic_content = gr.Slider(0.0, 10.0, value=1.5, label="Organic Content (%)")
        
        with gr.Column():
            gr.Markdown("### 🧪 NPK Nutrients & Weather")
            n = gr.Number(label="Nitrogen (N)", value=50)
            p = gr.Number(label="Phosphorus (P)", value=30)
            k = gr.Number(label="Potassium (K)", value=30)
            gr.Markdown("*Leave weather fields at 0 to automatically fetch live weather via Open-Meteo API*")
            temp = gr.Number(label="Temperature (°C)", value=0)
            humidity = gr.Number(label="Humidity (%)", value=0)
            rainfall = gr.Number(label="Rainfall (mm)", value=0)

    btn = gr.Button("Predict Crop Yield & Recommend Fertilizer 🚀", variant="primary")
    
    with gr.Row():
        output_yield = gr.Textbox(label="📈 Crop Yield Prediction", interactive=False)
        output_fert = gr.Textbox(label="💊 Fertilizer Prescription", interactive=False)

    btn.click(
        predict_crop_and_fertilizer,
        inputs=[state, crop, n, p, k, ph, moisture, organic_content, temp, humidity, rainfall],
        outputs=[output_yield, output_fert]
    )

demo.launch()
