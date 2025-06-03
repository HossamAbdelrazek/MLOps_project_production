from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
from fastapi.middleware.cors import CORSMiddleware
import logging
import joblib
from typing import List
import os
import time
import numpy as np
from preprocessing.preprocess import predict_gesture

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hand Gestures API")

# Initialize Prometheus metrics
# 1. MODEL-RELATED METRIC: Prediction confidence distribution
prediction_confidence = Histogram(
    'gesture_prediction_confidence',
    'Distribution of prediction confidence scores',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# 2. DATA-RELATED METRIC: Input data quality
invalid_input_counter = Counter(
    'invalid_input_requests_total',
    'Total number of requests with invalid input data',
    ['error_type']
)

# 3. SERVER-RELATED METRIC: Model loading status and API health
model_health_gauge = Gauge(
    'model_health_status',
    'Status of model and encoder loading (1=healthy, 0=unhealthy)'
)

# Additional useful metrics
gesture_predictions_counter = Counter(
    'gesture_predictions_total',
    'Total number of gesture predictions by type',
    ['gesture_type', 'direction']
)

prediction_latency = Histogram(
    'prediction_processing_time_seconds',
    'Time spent processing prediction requests',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Set up Prometheus instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/metrics"],
)
instrumentator.instrument(app).expose(app)

MODEL_PATH = "app/artifacts/model.pkl"
ENCODER_PATH = "app/artifacts/label_encoder.pkl"

# Load model and encoder at startup
model = None
encoder = None

def load_model_and_encoder():
    """Load model and encoder, update health metrics"""
    global model, encoder
    
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            encoder = joblib.load(ENCODER_PATH)
            model_health_gauge.set(1)  # Healthy
            logger.info("Model and encoder loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model or encoder: {e}")
            model = encoder = None
            model_health_gauge.set(0)  # Unhealthy
            return False
    else:
        logger.error(f"Model or encoder file not found at {MODEL_PATH} or {ENCODER_PATH}")
        model_health_gauge.set(0)  # Unhealthy
        return False

# Load models at startup
load_model_and_encoder()

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
    """Health check endpoint with model status"""
    model_loaded = model is not None
    encoder_loaded = encoder is not None
    
    # Update health gauge
    model_health_gauge.set(1 if (model_loaded and encoder_loaded) else 0)
    
    return {
        "status": "ok", 
        "model_loaded": model_loaded, 
        "encoder_loaded": encoder_loaded,
        "timestamp": time.time()
    }

@app.get("/health")
def detailed_health():
    """Detailed health check for monitoring"""
    model_loaded = model is not None
    encoder_loaded = encoder is not None
    
    return {
        "status": "healthy" if (model_loaded and encoder_loaded) else "unhealthy",
        "model_status": "loaded" if model_loaded else "not_loaded",
        "encoder_status": "loaded" if encoder_loaded else "not_loaded",
        "model_path_exists": os.path.exists(MODEL_PATH),
        "encoder_path_exists": os.path.exists(ENCODER_PATH),
        "timestamp": time.time()
    }

class GestureInput(BaseModel):
    data: List[float]  # Expects flattened list of 63 values (21 landmarks * 3 coordinates)

@app.post("/predict")
async def predict(gesture: GestureInput):
    """Predict gesture from hand landmark data"""
    start_time = time.time()
    
    # Check if model and encoder are loaded
    if model is None or encoder is None:
        invalid_input_counter.labels(error_type="model_not_loaded").inc()
        return None
    
    # Validate input data
    if not gesture.data:
        invalid_input_counter.labels(error_type="empty_data").inc()
        return None
        
    if len(gesture.data) != 63:
        invalid_input_counter.labels(error_type="invalid_length").inc()
        return None

    # Check for invalid numeric values
    try:
        data_array = np.array(gesture.data)
        if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
            invalid_input_counter.labels(error_type="invalid_numeric_values").inc()
            return None
    except Exception:
        invalid_input_counter.labels(error_type="data_conversion_error").inc()
        return None

    try:
        
        current_prediction, confidence = predict_gesture(model, gesture.data, encoder)
        
        # Record metrics
        prediction_confidence.observe(confidence)
        
        logger.info(f"Predicted gesture: {current_prediction} (confidence: {confidence:.3f})")
        
        # Only return valid gestures, otherwise return None
        if current_prediction in GESTURE_MAPPING:
            direction = GESTURE_MAPPING[current_prediction]
            gesture_predictions_counter.labels(
                gesture_type=current_prediction, 
                direction=direction
            ).inc()
            
            # Record processing time
            processing_time = time.time() - start_time
            prediction_latency.observe(processing_time)
            
            return direction
        else:
            invalid_input_counter.labels(error_type="unknown_gesture").inc()
            return None
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        invalid_input_counter.labels(error_type="prediction_exception").inc()
        return None

@app.get("/metrics-info")
def metrics_info():
    """Endpoint to describe available metrics for monitoring"""
    return {
        "metrics": {
            "model_related": {
                "gesture_prediction_confidence": "Distribution of prediction confidence scores",
                "gesture_predictions_total": "Total number of gesture predictions by type"
            },
            "data_related": {
                "invalid_input_requests_total": "Total number of requests with invalid input data"
            },
            "server_related": {
                "model_health_status": "Status of model and encoder loading (1=healthy, 0=unhealthy)",
                "prediction_processing_time_seconds": "Time spent processing prediction requests"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)