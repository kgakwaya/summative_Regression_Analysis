import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import SGDRegressor

# Initialize FastAPI App
app = FastAPI(
    title="Maternal Health Risk Prediction API",
    description="API for predicting Systolic Blood Pressure and triggering model retraining.",
    version="1.0.0",
    docs_url="/docs",  # Public Swagger UI endpoint
    redoc_url="/redoc"
)

# ==============================================================================
# CORS MIDDLEWARE CONFIGURATION & REASONING
# ==============================================================================
"""
CORS REASONING:
- allow_origins: We explicitly define permitted web domains/origins (such as our 
  production Flutter web app domain, local testing ports, and mobile web clients). 
  We strictly avoid using wildcards ("*") in production to prevent malicious third-party 
  websites from making unauthorized cross-origin requests to our API.
- allow_credentials: Set to True to allow authenticated cross-origin requests 
  (e.g., cookies, HTTP authentication, or authorization headers).
- allow_methods: Explicitly restricted to POST (for predictions/retraining), GET 
  (for status checks), and OPTIONS (for browser pre-flight requests).
- allow_headers: Restricted to essential headers like Content-Type and Authorization 
  to prevent header injection attacks.
"""

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1",
    "https://maternal-health-app.flutter.dev", # Example production Flutter app domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# ==============================================================================
# PATHS AND MODEL LOADING
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")

def load_artifacts():
    """Loads model and scaler artifacts from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or Scaler joblib file missing from directory.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# Load artifacts on startup
try:
    model, scaler = load_artifacts()
except Exception as e:
    model, scaler = None, None
    print(f"Warning: Artifacts could not be loaded on startup. Details: {e}")


# ==============================================================================
# PYDANTIC DATA MODELS (VALIDATION & RANGE CONSTRAINTS)
# ==============================================================================
class PredictionInput(BaseModel):
    Age: int = Field(
        ..., 
        ge=10, 
        le=70, 
        description="Age of the pregnant woman in years (10 - 70)", 
        example=25
    )
    DiastolicBP: float = Field(
        ..., 
        ge=40.0, 
        le=120.0, 
        description="Diastolic Blood Pressure in mmHg (40 - 120)", 
        example=80.0
    )
    BS: float = Field(
        ..., 
        ge=3.0, 
        le=20.0, 
        description="Blood Glucose Level in mmol/L (3.0 - 20.0)", 
        example=7.5
    )
    BodyTemp: float = Field(
        ..., 
        ge=95.0, 
        le=104.0, 
        description="Body Temperature in Fahrenheit (95.0 - 104.0)", 
        example=98.0
    )
    HeartRate: float = Field(
        ..., 
        ge=50.0, 
        le=120.0, 
        description="Heart Rate in bpm (50 - 120)", 
        example=70.0
    )

class PredictionOutput(BaseModel):
    predicted_systolic_bp: float
    status: str
    message: str

class RetrainInputRecord(BaseModel):
    Age: int = Field(..., ge=10, le=70)
    DiastolicBP: float = Field(..., ge=40.0, le=120.0)
    BS: float = Field(..., ge=3.0, le=20.0)
    BodyTemp: float = Field(..., ge=95.0, le=104.0)
    HeartRate: float = Field(..., ge=50.0, le=120.0)
    SystolicBP: float = Field(..., ge=70.0, le=200.0, description="Target label for retraining")

class RetrainBatchInput(BaseModel):
    data: List[RetrainInputRecord]


# ==============================================================================
# API ENDPOINTS
# ==============================================================================
@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return {
        "status": "online",
        "message": "Maternal Health Risk API is running. Go to /docs for Swagger UI documentation."
    }


@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
def predict(payload: PredictionInput):
    """
    Accepts maternal health features, standardizes them using fitted scaler,
    and returns predicted Systolic Blood Pressure.
    """
    global model, scaler
    if model is None or scaler is None:
        try:
            model, scaler = load_artifacts()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model or scaler artifacts are not available."
            )

    try:
        # Construct feature array matching training order
        features = np.array([[
            payload.Age,
            payload.DiastolicBP,
            payload.BS,
            payload.BodyTemp,
            payload.HeartRate
        ]])

        # Standardize using loaded scaler
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)[0]

        return PredictionOutput(
            predicted_systolic_bp=round(float(prediction), 2),
            status="success",
            message="Prediction generated successfully."
        )

    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction: {str(err)}"
        )


def execute_retraining(new_records: List[RetrainInputRecord]):
    """Background task function to retrain or update the model with new streaming data."""
    global model, scaler
    
    # Convert incoming array to DataFrame
    df_new = pd.DataFrame([r.dict() for r in new_records])
    
    X_new = df_new[["Age", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]]
    y_new = df_new["SystolicBP"]

    # Partial fit or re-fit model
    if hasattr(model, "partial_fit"):
        # Scale and incrementally update model if SGDRegressor
        X_scaled = scaler.transform(X_new)
        model.partial_fit(X_scaled, y_new)
    else:
        # Re-fit scaler and model on combined synthetic updates
        X_scaled = scaler.fit_transform(X_new)
        model.fit(X_scaled, y_new)

    # Save updated models to disk
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)


@app.post("/retrain", status_code=status.HTTP_202_ACCEPTED)
def retrain_model(batch: RetrainBatchInput, background_tasks: BackgroundTasks):
    """
    Triggers model retraining/updating with newly streamed data records.
    Runs asynchronously as a background task.
    """
    if len(batch.data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data provided for retraining."
        )

    background_tasks.add_task(execute_retraining, batch.data)

    return {
        "status": "accepted",
        "message": f"Retraining process triggered in background with {len(batch.data)} new data points."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("prediction:app", host="0.0.0.0", port=8000, reload=True)