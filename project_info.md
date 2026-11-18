# 🌾 AgriAI: Complete Project Study Guide & Reference Manual

Welcome to your offline study guide. This document contains standard explanations of the core theoretical and practical components of **AgriAI** so you can learn and research them from external sources (textbooks, online courses, and documentation).

---

## 📚 1. Core Machine Learning Concepts
To understand how AgriAI makes predictions, you should search and study these topics:

### A. Supervised Learning
* **What to search**: *"Supervised learning vs unsupervised learning"*, *"Regression vs Classification"*
* **How it applies here**: 
  * The **Crop Yield Predictor** is a **Regression** model because it predicts a continuous numerical target (`Yield_tons_per_ha`).
  * The **Fertilizer Recommender** is a **Classification** model because it predicts discrete categories/labels (`Fertilizer`).

### B. Decision Trees & Ensemble Learning
* **What to search**: *"Decision trees explained"*, *"Ensemble learning"*, *"Bagging vs Boosting"*
* **How it applies here**: Decision trees split data based on feature thresholds (e.g., *Is Nitrogen < 40?*). XGBoost is an **Ensemble** algorithm using **Gradient Boosting**, where trees are built sequentially to minimize the residuals (errors) of the previous trees.

### C. XGBoost (Extreme Gradient Boosting)
* **What to search**: *"XGBoost algorithm tutorial"*, *"Gradient Boosting machine"*
* **Why it's used**: XGBoost is highly optimized for performance on tabular datasets. It uses advanced regularization (L1/L2 penalties) to prevent overfitting and handles missing data natively.

---

## 🛠️ 2. Data Preprocessing Pipelines
Before feeding raw data (like text values for states or crops, and soil metrics with different scales) into XGBoost, we must preprocess it.

### A. Imputation
* **What to search**: *"Missing data imputation techniques in machine learning"*
* **How it applies here**: We use `SimpleImputer(strategy='median')` for numbers and `SimpleImputer(strategy='most_frequent')` for categories. If soil data has empty entries, it fills them with typical average values.

### B. Feature Scaling
* **What to search**: *"StandardScaler vs MinMaxScaler"*, *"Why feature scaling is important"*
* **How it applies here**: `StandardScaler` normalizes numeric features (like Temperature and Rainfall) so they have a mean of 0 and a standard deviation of 1. This prevents features with large ranges from dominating the model's loss calculations.

### C. Categorical Encoding
* **What to search**: *"One-hot encoding vs Label encoding"*
* **How it applies here**: ML models can only read numbers, not strings.
  * We use **OneHotEncoder** for input variables (State, Crop) to turn them into binary columns (e.g. `Crop_Wheat = 1`).
  * We use **LabelEncoder** to convert the output target class (Fertilizer name) into integer tokens (0, 1, 2, 3).

### D. Scikit-Learn Pipelines
* **What to search**: *"Scikit-learn pipeline tutorial"*, *"ColumnTransformer in scikit-learn"*
* **How it applies here**: We use `ColumnTransformer` to apply scaling to numerical data and one-hot encoding to categorical data, then bundle it with the XGBoost regressor/classifier into a unified `Pipeline`.

---

## 🌐 3. Web Services & APIs
AgriAI uses a decoupled architecture where the frontend and backend communicate via APIs.

### A. FastAPI & RESTful APIs
* **What to search**: *"FastAPI tutorial"*, *"REST API methods POST GET"*, *"Pydantic validation"*
* **How it applies here**: FastAPI hosts our ML models as endpoints. It uses Pydantic to validate that inputs (e.g. `pH` must be a float, `N` must be an integer) meet requirements before making predictions.

### B. Uvicorn
* **What to search**: *"What is ASGI"*, *"Uvicorn python server"*
* **How it applies here**: Uvicorn is the high-performance ASGI (Asynchronous Server Gateway Interface) web server used to run FastAPI.

---

## ☁️ 4. Dynamic Data & Deployment

### A. API Ingestion (Open-Meteo)
* **What to search**: *"JSON API integration in Python using requests"*
* **How it applies here**: We use the Python `requests` library to fetch current weather conditions (temp, humidity, rain) from Open-Meteo automatically when a user inputs coordinates/locations.

### B. Hugging Face Spaces & Gradio
* **What to search**: *"Hugging Face spaces documentation"*, *"Gradio blocks layout"*
* **How it applies here**: Gradio allows us to build simple machine learning web apps. Hugging Face hosts this app inside a sandboxed Docker container.

### C. ZeroGPU Architecture
* **What to search**: *"Hugging face spaces ZeroGPU"*
* **How it applies here**: ZeroGPU is Hugging Face's shared GPU system for free spaces. It requires `@spaces.GPU` decorators on Python functions to orchestrate resource allocation dynamically.
