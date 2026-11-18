import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor, XGBClassifier
import joblib
import os

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
        # We don't include weather for fertilizer, just soil + location + crop
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
