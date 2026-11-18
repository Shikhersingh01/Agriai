import pandas as pd
import numpy as np
import random
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Coordinates for North Indian States (Approximate centers)
LOCATION_COORDS = {
    "Punjab": {"lat": 31.1471, "lon": 75.3412},
    "Haryana": {"lat": 29.0588, "lon": 76.0856},
    "UP": {"lat": 26.8467, "lon": 80.9462},
    "Rajasthan": {"lat": 27.0238, "lon": 74.2179}
}

CROPS = ["Wheat", "Rice", "Sugarcane", "Mustard"]

def generate_proxy_data(n_samples=5000):
    """
    Generates synthetic but agronomically reasonable proxy data for North India.
    """
    np.random.seed(42)
    random.seed(42)

    data = []
    
    for _ in range(n_samples):
        state = random.choice(list(LOCATION_COORDS.keys()))
        crop = random.choice(CROPS)
        
        # Soil base parameters
        ph = round(random.uniform(5.5, 8.5), 2)
        organic_content = round(random.uniform(0.5, 2.5), 2)
        moisture = round(random.uniform(20.0, 60.0), 1)

        # Weather defaults (will vary by crop season roughly)
        if crop == "Wheat" or crop == "Mustard":
            # Rabi crops (Winter)
            temp = round(random.uniform(10.0, 25.0), 1)
            humidity = round(random.uniform(40.0, 70.0), 1)
            rainfall = round(random.uniform(10.0, 50.0), 1) 
        else:
            # Kharif / Cash crops (Summer/Monsoon)
            temp = round(random.uniform(25.0, 38.0), 1)
            humidity = round(random.uniform(60.0, 90.0), 1)
            rainfall = round(random.uniform(100.0, 400.0), 1)
            
        # NPK values driven mostly by soil ph and organic content realistically, but randomized with bounds
        n = random.randint(20, 120)
        p = random.randint(10, 60)
        k = random.randint(10, 60)

        # Yield formulation (fake baseline but logical)
        base_yield = {"Wheat": 3.5, "Rice": 4.0, "Sugarcane": 70.0, "Mustard": 1.5}[crop]
        # Modifiers
        ph_mod = 1.0 - abs(7.0 - ph) * 0.1
        water_mod = 1.0 if (moisture + rainfall > 100) else 0.8
        
        yield_val = base_yield * ph_mod * water_mod * random.uniform(0.8, 1.2)
        yield_val = round(max(0.1, yield_val), 2)

        # Fertilizer recommendation logic (simple proxy)
        if n < 40:
            fert = "Urea High Dose"
        elif n < 80 and p < 30:
            fert = "DAP (Di-ammonium Phosphate)"
        elif ph < 6.0:
            fert = "NPK 19:19:19 + Lime"
        else:
            fert = "Standard NPK 12:32:16"

        data.append({
            "State": state,
            "Crop": crop,
            "N": n,
            "P": p,
            "K": k,
            "pH": ph,
            "Moisture": moisture,
            "Organic_Content": organic_content,
            "Temperature": temp,
            "Humidity": humidity,
            "Rainfall": rainfall,
            "Yield_tons_per_ha": yield_val,
            "Fertilizer": fert
        })
        
    df = pd.DataFrame(data)
    return df

def fetch_weather_by_location(state_name: str):
    """
    Smart feature: fetches current/historical average weather from Open-Meteo.
    Used if user leaves weather fields blank in UI.
    """
    if state_name not in LOCATION_COORDS:
        logger.warning(f"State {state_name} not found. Defaulting to UP.")
        state_name = "UP"
        
    coords = LOCATION_COORDS[state_name]
    lat, lon = coords["lat"], coords["lon"]
    
    try:
        # Fetching current weather as a proxy for the input
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        
        return {
            "Temperature": current.get("temperature_2m", 25.0),
            "Humidity": current.get("relative_humidity_2m", 60.0),
            "Rainfall": current.get("precipitation", 0.0) # Using precip as daily rainfall context 
        }
    except Exception as e:
        logger.error(f"Failed to fetch weather from API: {e}")
        # Fallback averages
        return {
            "Temperature": 28.0,
            "Humidity": 55.0,
            "Rainfall": 5.0
        }

if __name__ == "__main__":
    import os
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    df = generate_proxy_data(5000)
    df.to_csv(os.path.join(os.path.dirname(__file__), 'crop_data_proxy.csv'), index=False)
    print("Proxy dataset generated at crop_data_proxy.csv")
