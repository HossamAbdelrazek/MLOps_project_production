from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware
import logging
import joblib
from typing import List
import os
from app.preprocessing.preprocess import predict_gesture

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hand Gestures API")
Instrumentator().instrument(app).expose(app)

MODEL_PATH = "app/artifacts/model.pkl"
ENCODER_PATH = "app/artifacts/label_encoder.pkl"

# Load model and encoder at startup
model = None
encoder = None

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        logger.info("Model and encoder loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model or encoder: {e}")
        model = encoder = None
else:
    logger.error(f"Model or encoder file not found at {MODEL_PATH} or {ENCODER_PATH}")

# Gesture mapping for game controls
GESTURE_MAPPING = {
    "like": "up",
    "dislike": "down", 
    "peace": "left",
    "peace_inverted": "right"
}

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],  # Live server origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok", "model_loaded": model is not None, "encoder_loaded": encoder is not None}

class GestureInput(BaseModel):
    data: List[float]  # Expects flattened list of 63 values (21 landmarks * 3 coordinates)

@app.post("/predict")
async def predict(gesture: GestureInput):
    if model is None or encoder is None:
        return None
    
    if not gesture.data or len(gesture.data) != 63:
        return None

    try:
        current_prediction, confidence = predict_gesture(model, gesture.data, encoder)
        logger.info(f"Predicted gesture: {current_prediction}, Confidence: {confidence:.2f}")
        
        # Only return valid gestures, otherwise return None
        if current_prediction in GESTURE_MAPPING:
            return GESTURE_MAPPING[current_prediction]
        else:
            return None
            
    except Exception as e:
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)