"""
AIML Module for Focus/Attention Tracking System

This module provides:
- Feature engineering for posture and tab switching events
- Behavior model for distraction prediction
- Session model for focus scoring
- Easy-to-use inference interfaces
"""

__version__ = "1.0.0"

from aiml.inference.predictor import FocusPredictor
from aiml.inference.session_scorer import SessionScorer
from aiml.models.feature_engineering import engineer_features

__all__ = [
    "FocusPredictor",
    "SessionScorer",
    "engineer_features"
]

