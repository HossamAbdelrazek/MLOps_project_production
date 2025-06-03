# Hand Gestures Recognition API

This repository contains a **FastAPI-based** web service for **hand gesture recognition**. It predicts gestures from landmark data and maps them to control commands like `"up"`, `"down"`, `"left"`, and `"right"`. The project includes a preprocessing pipeline, trained models, monitoring with Prometheus, and Docker support for easy deployment.

---

## 📂 Project Structure

```
production/
├── app/
│   └── __pycache__/
├── artifacts/
│   ├── label_encoder.pkl
│   └── model.pkl
├── preprocessing/
│   ├── __init__.py
│   └── preprocess.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── dockerfile
├── docker-compose.yaml
├── main.py
├── requirements.txt
├── monitoring/
└── venv/
```

---

## 🚀 Features

- Gesture Recognition: Predict hand gestures from landmark data and map them to actions (up, down, left, right).

- Preprocessing Module: Encapsulates the logic for preparing inputs and running model inference.

- FastAPI Server: Provides /predict and / endpoints.

- Monitoring: Prometheus metrics are integrated using prometheus-fastapi-instrumentator.

- Docker-Ready: Includes dockerfile and docker-compose.yaml for containerized deployment.

- CORS Support: Configured to allow requests from specified frontend origins. 

---

## 📝 Usage

### 🔧 Local Setup

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd production
   ```

2. Create a virtual environment and install dependencies

   ```bash
   python3 -m venv venv 
   
   source venv/bin/activate
   
   cd app/

   pip install -r app/requirements.txt
   ```

3. Run the FastAPI server

   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8080
   ```

## 🐳 Docker Setup

1. Build and run the containers

   ```bash
   docker-compose up --build
   ```

2. Access the FastAPI server at http://localhost:8080

## 📬 API Endpoints

- GET /
Health check. Returns the server status and model loading state.

{
  "status": "ok",
  "model_loaded": true,
  "encoder_loaded": true
}

- POST /predict
Accepts a JSON body with a list of 63 float values (21 landmarks × 3 coordinates).
Example:
{
  "data": [0.1, 0.2, ..., 0.3]
}

Returns one of: "up", "down", "left", "right", or null if unrecognized.

## 📈 Monitoring

- Prometheus metrics are automatically exposed for monitoring the API's performance and usage.

## 🐳 Dockerized Deployment

- dockerfile: Builds the image with FastAPI and dependencies.

- docker-compose.yaml: Simplifies running the API container along with potential Prometheus/Grafana monitoring (if configured in monitoring/).

## 🌐 CORS Configuration

CORS middleware is configured to allow frontend requests from:

- http://localhost:5500

- http://127.0.0.1:5500

You can modify these origins in main.py to match your frontend deployment.

## 💡 Example Prediction Request with curl

   ```bash
   curl -X POST "http://localhost:8080/predict" \
      -H "Content-Type: application/json" \
      -d '{"data": [0.1, 0.2, ..., 0.3]}'
   ```

## 📸 Screenshots

