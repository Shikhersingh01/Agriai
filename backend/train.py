import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import logging
from data import generate_proxy_data
from model import CropPredictorModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_save():
    # 1. Generate Data
    logger.info("Generating proxy dataset for North India...")
    df = generate_proxy_data(5000)
    
    # Save dataset to disk for review
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, 'crop_data_proxy.csv'), index=False)

    # 2. Split Data for Yield Model
    logger.info("Training Yield Model...")
    X_yield = df.drop(columns=['Yield_tons_per_ha', 'Fertilizer'])
    y_yield = df['Yield_tons_per_ha']
    
    X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X_yield, y_yield, test_size=0.2, random_state=42)

    # 3. Train Yield Model
    predictor = CropPredictorModel()
    yield_model = predictor.build_yield_pipeline()
    yield_model.fit(X_train_y, y_train_y)
    
    # Evaluate Yield Model
    y_pred_y = yield_model.predict(X_test_y)
    rmse = mean_squared_error(y_test_y, y_pred_y) ** 0.5
    r2 = r2_score(y_test_y, y_pred_y)
    logger.info(f"Yield Model Evaluation -> RMSE: {rmse:.3f}, R2 Score: {r2:.3f}")

    # 4. Split Data for Fertilizer Model
    logger.info("Training Fertilizer Recommendation Model...")
    # We exclude weather from fertilizer recommendation as it's primarily soil-dependent in our logic
    X_fert = df.drop(columns=['Yield_tons_per_ha', 'Fertilizer', 'Temperature', 'Humidity', 'Rainfall'])
    
    y_fert_encoded = predictor.fert_label_encoder.fit_transform(df['Fertilizer'])
    
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fert, y_fert_encoded, test_size=0.2, random_state=42)
    
    # 5. Train Fertilizer Model
    fert_model = predictor.build_fert_pipeline()
    fert_model.fit(X_train_f, y_train_f)
    
    # Evaluate Fertilizer Model
    y_pred_f = fert_model.predict(X_test_f)
    acc = accuracy_score(y_test_f, y_pred_f)
    logger.info(f"Fertilizer Model Accuracy: {acc:.3f}")
    
    # 6. Save Models
    models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(yield_model, os.path.join(models_dir, 'yield_model.pkl'))
    joblib.dump(fert_model, os.path.join(models_dir, 'fert_model.pkl'))
    joblib.dump(predictor.fert_label_encoder, os.path.join(models_dir, 'fert_label_encoder.pkl'))
    
    logger.info(f"Models successfully saved to {models_dir}")

if __name__ == "__main__":
    train_and_save()
