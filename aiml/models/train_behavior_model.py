"""
Train Behavior Model for Distraction Prediction

Trains XGBoost model on event data (posture + tab switching).
"""
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))

from aiml.models.feature_engineering import engineer_features

# Configuration
DATA_PATH = current_dir.parent / "data" / "events.csv"
MODEL_PATH = current_dir / "behavior_model.pkl"

def main():
    print("=" * 60)
    print("Training Behavior Model (XGBoost)")
    print("=" * 60)
    
    # Check if data file exists
    if not DATA_PATH.exists():
        print(f"\n[ERROR] Data file not found: {DATA_PATH}")
        print("Please ensure events.csv exists in aiml/data/")
        return
    
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} events")
    
    # Engineer features
    print("\nEngineering features...")
    X, y = engineer_features(df)
    print(f"Created {len(X.columns)} features")
    print(f"Features: {list(X.columns)}")
    
    # Split data
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print("\nTraining XGBoost model...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"\n[RESULTS]")
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Save model
    print(f"\nSaving model to: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    print("[SUCCESS] Model saved!")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

