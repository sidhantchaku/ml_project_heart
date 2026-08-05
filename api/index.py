from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Get the absolute path to the root directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(base_dir, "heart_model.pkl")
SCALER_PATH = os.path.join(base_dir, "scaler.pkl")

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    model = None
    scaler = None
    print(f"Error loading model: {e}")

class PatientData(BaseModel):
    age: int
    sex: str
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/api/predict")
def predict(data: PatientData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded correctly.")

    sex_num = 1 if data.sex == "Male" else 0
    input_data = np.array([[
        data.age, sex_num, data.cp, data.trestbps, data.chol, data.fbs, 
        data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, 
        data.ca, data.thal
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = int(model.predict(input_scaled)[0])
    prob = float(model.predict_proba(input_scaled)[0][1])

    return {
        "prediction": prediction,
        "probability": prob
    }

@app.get("/api/health")
def health():
    return {"status": "healthy"}
