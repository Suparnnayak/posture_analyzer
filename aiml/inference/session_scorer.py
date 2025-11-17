"""
Session Scorer - Focus Score Prediction for Complete Sessions

This module provides focus scoring for completed sessions.
"""
import joblib
import pandas as pd
import sys
from pathlib import Path

# Handle imports - works both standalone and as module
try:
    pass  # No imports needed here
except ImportError:
    pass


class SessionScorer:
    """
    Scores focus level for completed sessions.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the session scorer.
        
        Args:
            model_path: Path to trained model file. If None, uses default path.
        """
        if model_path is None:
            # Default to models directory
            current_dir = Path(__file__).parent.parent
            model_path = current_dir / "models" / "session_model.pkl"
            # If running from within aiml/, adjust path
            if not model_path.exists() and Path("models/session_model.pkl").exists():
                model_path = Path("models/session_model.pkl")
        
        # Convert to absolute path
        model_path = Path(model_path).resolve()
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please train the model first using: python aiml/models/train_session_model.py"
            )
        
        self.model = joblib.load(model_path)
        self.model_path = model_path
    
    def score_session(self, session_data: dict) -> dict:
        """
        Score a session's focus level.
        
        Expected input format:
        {
            "avg_slouch": 0.3,           # average slouch frequency
            "avg_lookaway": 0.2,         # average lookaway frequency
            "total_breaks": 3,           # number of breaks
            "total_break_time": 120.0,   # total break time in seconds
            "session_length": 1500.0,    # session length in seconds
            "total_tab_switches": 15,    # total tab switches (optional)
            "avg_tab_frequency": 0.1      # average tab switch frequency (optional)
        }
        
        Returns:
            {
                "focus_score": 75.5  # focus score (0-100, higher = more focused)
            }
        """
        try:
            # Prepare data
            df = pd.DataFrame([session_data])
            
            # Ensure required columns exist
            required_cols = [
                "avg_slouch",
                "avg_lookaway",
                "total_breaks",
                "total_break_time",
                "session_length"
            ]
            
            # Add optional tab switching features if not present
            if "total_tab_switches" not in df.columns:
                df["total_tab_switches"] = 0
            if "avg_tab_frequency" not in df.columns:
                df["avg_tab_frequency"] = 0.0
            
            # Select features (model expects specific columns)
            feature_cols = [
                "avg_slouch",
                "avg_lookaway",
                "total_breaks",
                "total_break_time",
                "session_length"
            ]
            
            # Use only the features the model was trained on
            X = df[feature_cols]
            
            # Predict
            score = float(self.model.predict(X)[0])
            
            # Ensure score is in reasonable range (0-100)
            score = max(0, min(100, score))
            
            return {"focus_score": score}
        except Exception as e:
            # Return safe default on error
            return {
                "focus_score": 50.0,
                "error": str(e)
            }

