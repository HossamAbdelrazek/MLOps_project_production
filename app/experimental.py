from preprocessing import predict_gesture
import pandas as pd
from typing import Literal
import logging
import joblib
import cv2
import mediapipe as mp
import numpy as np
import os
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logged_model = "app/artifacts/LGBM.pkl"
model = joblib.load(logged_model) if os.path.exists(logged_model) else None
logger.info(f"Loading model") if model else logger.error("Model not found")

app = FastAPI(title="Hand Gestures API")

Instrumentator().instrument(app).expose(app)

ENCODER = "artifacts/label_encoder.pkl"

def load_encoder(path=ENCODER):
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Encoder file not found at: {path}")
    encoder = joblib.load(path)
    return encoder

encoder = load_encoder()

os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Load model
model = joblib.load("MLPack/model.joblib")
encoder = joblib.load("MLPack/encoder.joblib")

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
K = 3

def get_hand_gestures():
      # Number of frames to accumulate for prediction
    cap = cv2.VideoCapture(0)

    # Video Writer setup to record video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('hand_gesture_output.avi', fourcc, 10, (640, 480)) 

    frame_buffer = []
    current_prediction = None
    current_confidence = 0.0

    cv2.namedWindow("Hand_gestures_Application", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand_gestures_Application", 1280, 720)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as hands:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            landmarks = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])

                frame_buffer.append(landmarks)
                if len(frame_buffer) == K:
                    current_prediction, _ = predict_gesture(model, frame_buffer, encoder)
                    frame_buffer.pop(0)
                else:
                    current_confidence = None

            # MAIN FRAME (GAME/APP WINDOW)
            h_main, w_main = 720, 1280
            main_display = np.zeros((h_main, w_main, 3), dtype=np.uint8)  # Black background for game

            # Simulate the game window (could overlay game graphics here)
            # Here we just keep the original webcam as a placeholder for the game background.
            game_frame_resized = cv2.resize(frame, (w_main, h_main))
            main_display[:h_main, :w_main] = game_frame_resized

            # RESIZE FRAME FOR SMALL VIDEO
            small_h, small_w = 240, 320  # Size for small video
            small_frame = cv2.resize(frame, (small_w, small_h))

            # OVERLAY SMALL FRAME IN BOTTOM-RIGHT
            y_offset = h_main - small_h - 10  # 10px margin
            x_offset = w_main - small_w - 10
            main_display[y_offset:y_offset+small_h, x_offset:x_offset+small_w] = small_frame

            # DISPLAY WINDOW
            cv2.imshow("Hand_gestures_Application", main_display)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return current_prediction

class Hands_data(BaseModel):
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    x3: float
    y3: float
    z3: float
    x4: float
    y4: float
    z4: float
    x5: float
    y5: float
    z5: float
    x6: float
    y6: float
    z6: float
    x7: float
    y7: float
    z7: float
    x8: float
    y8: float
    z8: float
    x9: float
    y9: float
    z9: float
    x10: float
    y10: float
    z10: float
    x11: float
    y11: float
    z11: float
    x12: float
    y12: float
    z12: float
    x13: float
    y13: float
    z13: float
    x14: float
    y14: float
    z14: float
    x15: float
    y15: float
    z15: float
    x16: float
    y16: float
    z16: float
    x17: float
    y17: float
    z17: float
    x18: float
    y18: float
    z18: float
    x19: float
    y19: float
    z19: float
    x20: float
    y20: float
    z20: float
    x21: float
    y21: float
    z21: float


@app.get("/")
def health():
    logger.info("Health check endpoint called")
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Hands_data):
    try:
        prediction = get_hand_gestures()
        result = int(prediction[0])

        logger.info(f"Prediction result: {result}")
        return {"churn_prediction": result}

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return HTTPException(status_code=500, detail=e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    