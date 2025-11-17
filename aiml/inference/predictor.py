"""
Focus Predictor - Distraction Prediction for Single Events

This module provides real-time distraction prediction for individual events.
"""
import joblib
import pandas as pd
import os
import sys
from pathlib import Path

# Handle imports - works both standalone and as module
try:
    from aiml.models.feature_engineering import engineer_features, prepare_single_event
except ImportError:
    # Running from within aiml/ directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.feature_engineering import engineer_features, prepare_single_event


class FocusPredictor:
    """
    Predicts distraction probability for individual events.
    
    Supports both posture and tab switching events.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model file. If None, uses default path.
        """
        if model_path is None:
            # Default to models directory
            current_dir = Path(__file__).parent.parent
            model_path = current_dir / "models" / "behavior_model.pkl"
            # If running from within aiml/, adjust path
            if not model_path.exists() and Path("models/behavior_model.pkl").exists():
                model_path = Path("models/behavior_model.pkl")
        
        # Convert to absolute path
        model_path = Path(model_path).resolve()
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please train the model first using: python aiml/models/train_behavior_model.py"
            )
        
        self.model = joblib.load(model_path)
        self.model_path = model_path
    
    def predict(self, raw_event: dict) -> dict:
        """
        Predict distraction probability for a single event.
        
        Expected input format:
        {
            "timestamp": 1731652010000,  # milliseconds
            "user_id": 1,
            "spineAngle": 22.5,          # for posture events
            "slouch": 1,                  # 0 or 1
            "lookingAway": 0,             # 0 or 1
            "eventType": "posture",        # or "tab_switch", "break_start", etc.
            "duration": 0,
            "tabCount": 0,                # for tab_switch events
            "tabFrequency": 0.0           # for tab_switch events
        }
        
        Returns:
            {
                "distraction_prob": 0.63,  # probability of being distracted
                "focus_prob": 0.37,         # probability of being focused
                "is_distracted": true         # binary prediction
            }
        """
        try:
            # Prepare event for feature engineering
            df = prepare_single_event(raw_event)
            
            # Engineer features
            X, _ = engineer_features(df)
            
            # Predict
            prob = float(self.model.predict_proba(X)[0][1])
            
            return {
                "distraction_prob": prob,
                "focus_prob": 1 - prob,
                "is_distracted": prob > 0.5
            }
        except Exception as e:
            # Return safe defaults on error
            return {
                "distraction_prob": 0.5,
                "focus_prob": 0.5,
                "is_distracted": False,
                "error": str(e)
            }

