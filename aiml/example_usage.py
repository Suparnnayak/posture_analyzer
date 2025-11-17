"""
Example Usage of AIML Module

This script demonstrates how to use the AIML module for:
1. Predicting distraction from posture events
2. Predicting distraction from tab switching events
3. Scoring complete sessions
"""
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from inference.predictor import FocusPredictor
from inference.session_scorer import SessionScorer

def main():
    print("=" * 60)
    print("AIML Module - Example Usage")
    print("=" * 60)
    
    # Initialize predictors
    print("\n[1] Initializing predictors...")
    try:
        predictor = FocusPredictor()
        scorer = SessionScorer()
        print("[OK] Predictors initialized")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Please train models first: python models/train_behavior_model.py")
        return
    
    # Example 1: Posture event
    print("\n[2] Example: Posture Event")
    print("-" * 60)
    posture_event = {
        "timestamp": 1731652010000,
        "user_id": 1,
        "spineAngle": 25.5,
        "slouch": 1,
        "lookingAway": 0,
        "eventType": "posture",
        "duration": 0
    }
    
    print("Input event:")
    for key, value in posture_event.items():
        print(f"  {key}: {value}")
    
    result = predictor.predict(posture_event)
    print("\nPrediction result:")
    print(f"  Distraction Probability: {result['distraction_prob']:.2%}")
    print(f"  Focus Probability: {result['focus_prob']:.2%}")
    print(f"  Is Distracted: {result['is_distracted']}")
    
    # Example 2: Tab switch event
    print("\n[3] Example: Tab Switch Event")
    print("-" * 60)
    tab_event = {
        "timestamp": 1731652011000,
        "user_id": 1,
        "eventType": "tab_switch",
        "duration": 0,
        "tabCount": 5,
        "tabFrequency": 0.1,
        "spineAngle": 0,
        "slouch": 0,
        "lookingAway": 0
    }
    
    print("Input event:")
    for key, value in tab_event.items():
        print(f"  {key}: {value}")
    
    result = predictor.predict(tab_event)
    print("\nPrediction result:")
    print(f"  Distraction Probability: {result['distraction_prob']:.2%}")
    print(f"  Focus Probability: {result['focus_prob']:.2%}")
    print(f"  Is Distracted: {result['is_distracted']}")
    
    # Example 3: Session scoring
    print("\n[4] Example: Session Scoring")
    print("-" * 60)
    session_data = {
        "avg_slouch": 0.3,
        "avg_lookaway": 0.2,
        "total_breaks": 3,
        "total_break_time": 120.0,
        "session_length": 1500.0
    }
    
    print("Session data:")
    for key, value in session_data.items():
        print(f"  {key}: {value}")
    
    score_result = scorer.score_session(session_data)
    print(f"\nFocus Score: {score_result['focus_score']:.1f}/100")
    
    # Example 4: Multiple events
    print("\n[5] Example: Processing Multiple Events")
    print("-" * 60)
    events = [
        {"timestamp": 1731652010000, "user_id": 1, "spineAngle": 20.0, "slouch": 0, "lookingAway": 0, "eventType": "posture", "duration": 0},
        {"timestamp": 1731652011000, "user_id": 1, "spineAngle": 28.5, "slouch": 1, "lookingAway": 0, "eventType": "posture", "duration": 0},
        {"timestamp": 1731652012000, "user_id": 1, "eventType": "tab_switch", "duration": 0, "tabCount": 3, "tabFrequency": 0.15, "spineAngle": 0, "slouch": 0, "lookingAway": 0},
    ]
    
    print(f"Processing {len(events)} events...")
    for i, event in enumerate(events, 1):
        result = predictor.predict(event)
        print(f"  Event {i}: Distraction = {result['distraction_prob']:.2%}, Is Distracted = {result['is_distracted']}")
    
    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nYou can now integrate this into your backend:")
    print("  from aiml.inference.predictor import FocusPredictor")
    print("  predictor = FocusPredictor()")
    print("  result = predictor.predict(event_dict)")

if __name__ == "__main__":
    main()

