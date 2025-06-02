import numpy as np
import pandas as pd
import logging

def normalize_hand_landmarks(landmarks):
    """Normalize hand landmarks by centering around (x1, y1) and scaling by (y13)."""
    landmarks = np.array(landmarks)
    
    # Extract x, y, z coordinates
    xs = landmarks[:, 0]
    ys = landmarks[:, 1] 
    zs = landmarks[:, 2]
    
    # Center around (x1, y1) - subtract x1, y1 from all coordinates
    x1, y1 = xs[0], ys[0]
    xs_norm = xs - x1
    ys_norm = ys - y1
    
    # Scale using y13 (13th point, index 12)
    y13 = ys_norm[12]  # This is y13 after centering
    xs_norm = xs_norm / y13
    ys_norm = ys_norm / y13
    
    # Create feature dictionary (like your DataFrame columns)
    features = {}
    for i in range(21):
        features[f'x{i+1}'] = xs_norm[i]
        features[f'y{i+1}'] = ys_norm[i]
        features[f'z{i+1}'] = zs[i]
    
    # Remove x1, y1 (they become 0 after centering, so the model doesn't need them)
    features.pop('x1', None)
    features.pop('y1', None)
    
    # Return as ordered array (sorted by feature names for consistency)
    feature_names = sorted(features.keys())
    return [features[name] for name in feature_names]

def predict_gesture(model, frame, encoder):
    """
    Predict gesture from a single frame (list of floats representing hand landmarks).
    """
    try:
        # Convert frame to (21, 3) array
        landmarks = np.array(frame).reshape((21, 3))
        
        # Normalize landmarks
        normalized_features = normalize_hand_landmarks(landmarks)
        
        # Convert to DataFrame for prediction
        input_data = pd.DataFrame([normalized_features])
        
        # Get predictions and probabilities
        prediction = model.predict(input_data)
        
        decoded_prediction = encoder.inverse_transform([prediction])
        logging.info(f"Predicted gesture: {decoded_prediction}")
        return decoded_prediction
        
    except Exception as e:
        print(f"Error in predict_gesture: {e}")
        return "unknown", 0.0