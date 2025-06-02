import numpy as np
import pandas as pd

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
    
    # Create feature dictionary based on model's expected features
    features = {}
    for i in range(21):
        features[f'x{i+1}'] = xs_norm[i]
        features[f'y{i+1}'] = ys_norm[i]
        features[f'z{i+1}'] = zs[i]
    
    # Based on model feature names, remove x1 and x2
    features.pop('x1', None)
    features.pop('x2', None)
    
    # Return features in the exact order the model expects
    model_feature_order = ['y1', 'z1', 'y2', 'z2']
    for i in range(3, 22):  # x3 to x21, y3 to y21, z3 to z21
        model_feature_order.extend([f'x{i}', f'y{i}', f'z{i}'])
    
    return [features[name] for name in model_feature_order]

def predict_gesture(model, frame, encoder):
    """
    Predict gesture from a single frame (list of floats representing hand landmarks).
    """
    try:
        # Convert frame to (21, 3) array
        landmarks = np.array(frame).reshape((21, 3))
        
        # Normalize landmarks
        normalized_features = normalize_hand_landmarks(landmarks)
        
        # Create feature names in the EXACT order the model expects
        # Pattern: y1, z1, y2, z2, x3, y3, z3, x4, y4, z4, ..., x21, y21, z21
        feature_names = ['y1', 'z1', 'y2', 'z2']
        for i in range(3, 22):  # Points 3 to 21
            feature_names.extend([f'x{i}', f'y{i}', f'z{i}'])
        
        # Convert to DataFrame with proper feature names
        input_data = pd.DataFrame([normalized_features], columns=feature_names)
        
        # Get predictions and probabilities
        predictions = model.predict(input_data)
        class_probabilities = model.predict_proba(input_data)
        
        prediction = predictions[0]  # This should be a single value, not array
        confidence = np.max(class_probabilities[0])
        
        # Fix encoder input - it expects single value, not array
        decoded_prediction = encoder.inverse_transform([prediction])[0]
        return decoded_prediction, confidence
        
    except Exception as e:
        print(f"Error in predict_gesture: {e}")
        print(f"Model feature names: {model.feature_names_in_ if hasattr(model, 'feature_names_in_') else 'Not available'}")
        return "unknown", 0.0