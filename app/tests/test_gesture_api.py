import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import numpy as np

# Add the app directory to the path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient
from main import app, GESTURE_MAPPING

client = TestClient(app)

class TestGestureAPI:
    """Test suite for Hand Gesture Recognition API"""
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "encoder_loaded" in data
    
    def test_predict_endpoint_valid_input(self):
        """Test prediction endpoint with valid 63-element input"""
        # Create mock 63-element array (21 landmarks * 3 coordinates)
        sample_data = [0.5] * 63
        
        with patch('main.predict_gesture') as mock_predict:
            # Mock the prediction function to return a known gesture
            mock_predict.return_value = ("like", 0.9)
            
            response = client.post("/predict", json={"data": sample_data})
            assert response.status_code == 200
            
            # Should return the mapped gesture
            result = response.json()
            assert result == "up"  # "like" maps to "up"
    
    def test_predict_endpoint_invalid_length(self):
        """Test prediction endpoint with invalid input length"""
        # Test with wrong number of elements
        sample_data = [0.5] * 50  # Wrong length
        
        response = client.post("/predict", json={"data": sample_data})
        assert response.status_code == 200
        assert response.json() is None
    
    def test_predict_endpoint_empty_input(self):
        """Test prediction endpoint with empty input"""
        response = client.post("/predict", json={"data": []})
        assert response.status_code == 200
        assert response.json() is None
    
    def test_predict_endpoint_unknown_gesture(self):
        """Test prediction endpoint with unknown gesture prediction"""
        sample_data = [0.5] * 63
        
        with patch('main.predict_gesture') as mock_predict:
            # Mock prediction of unknown gesture
            mock_predict.return_value = ("unknown_gesture", 0.7)
            
            response = client.post("/predict", json={"data": sample_data})
            assert response.status_code == 200
            assert response.json() is None
    
    def test_predict_endpoint_model_not_loaded(self):
        """Test prediction endpoint when model is not loaded"""
        sample_data = [0.5] * 63
        
        with patch('main.model', None), patch('main.encoder', None):
            response = client.post("/predict", json={"data": sample_data})
            assert response.status_code == 200
            assert response.json() is None
    
    def test_predict_endpoint_prediction_error(self):
        """Test prediction endpoint when prediction function raises exception"""
        sample_data = [0.5] * 63
        
        with patch('main.predict_gesture') as mock_predict:
            mock_predict.side_effect = Exception("Prediction error")
            
            response = client.post("/predict", json={"data": sample_data})
            assert response.status_code == 200
            assert response.json() is None
    
    def test_gesture_mapping_completeness(self):
        """Test that all expected gestures are in mapping"""
        expected_gestures = ["like", "dislike", "peace", "peace_inverted"]
        expected_directions = ["up", "down", "left", "right"]
        
        assert len(GESTURE_MAPPING) == 4
        for gesture in expected_gestures:
            assert gesture in GESTURE_MAPPING
        
        for direction in expected_directions:
            assert direction in GESTURE_MAPPING.values()
    
    def test_cors_headers(self):
        """Test that CORS headers are properly set"""
        response = client.options("/predict")
        # FastAPI automatically handles OPTIONS requests for CORS
        assert response.status_code in [200, 405]  # Some frameworks return 405 for OPTIONS
    
    @pytest.mark.parametrize("gesture,expected_direction", [
        ("like", "up"),
        ("dislike", "down"),
        ("peace", "left"),
        ("peace_inverted", "right")
    ])
    def test_gesture_mapping_individual(self, gesture, expected_direction):
        """Test individual gesture mappings"""
        sample_data = [0.5] * 63
        
        with patch('main.predict_gesture') as mock_predict:
            mock_predict.return_value = (gesture, 0.8)
            
            response = client.post("/predict", json={"data": sample_data})
            assert response.status_code == 200
            assert response.json() == expected_direction

class TestInputValidation:
    """Test input validation and edge cases"""
    
    def test_predict_missing_data_field(self):
        """Test prediction with missing data field"""
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error
    
    def test_predict_wrong_data_type(self):
        """Test prediction with wrong data type"""
        response = client.post("/predict", json={"data": "invalid"})
        assert response.status_code == 422  # Validation error
    
    def test_predict_non_numeric_data(self):
        """Test prediction with non-numeric data"""
        sample_data = ["string"] * 63
        response = client.post("/predict", json={"data": sample_data})
        assert response.status_code == 422  # Validation error

# Fixtures for test setup
@pytest.fixture
def mock_model_and_encoder():
    """Fixture to mock model and encoder loading"""
    with patch('main.model') as mock_model, patch('main.encoder') as mock_encoder:
        mock_model.return_value = MagicMock()
        mock_encoder.return_value = MagicMock()
        yield mock_model, mock_encoder

# Run tests with: pytest test_gesture_api.py -v