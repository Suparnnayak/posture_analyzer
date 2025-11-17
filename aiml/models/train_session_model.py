"""
Train Session Model for Focus Scoring

Trains Random Forest model on session summary data.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))

# Configuration
DATA_PATH = current_dir.parent / "data" / "sessions.csv"
MODEL_PATH = current_dir / "session_model.pkl"

def main():
    print("=" * 60)
    print("Training Session Scoring Model (Random Forest)")
    print("=" * 60)
    
    # Check if data file exists
    if not DATA_PATH.exists():
        print(f"\n[ERROR] Data file not found: {DATA_PATH}")
        print("Please ensure sessions.csv exists in aiml/data/")
        return
    
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} sessions")
    
    # Feature columns for session scoring
    feature_cols = [
        "avg_slouch",
        "avg_lookaway",
        "total_breaks",
        "total_break_time",
        "session_length"
    ]
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"\n[ERROR] Missing required columns: {missing_cols}")
        return
    
    X = df[feature_cols]
    y = df["focus_score"]
    
    print(f"\nFeatures: {feature_cols}")
    print(f"Target: focus_score")
    print(f"Data shape: {X.shape}")
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # Evaluate
    train_score = model.score(X, y)
    predictions = model.predict(X)
    mae = abs(predictions - y).mean()
    rmse = ((predictions - y) ** 2).mean() ** 0.5
    
    print(f"\n[RESULTS]")
    print(f"R² Score: {train_score:.4f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    
    # Save model
    print(f"\nSaving model to: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    print("[SUCCESS] Model saved!")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

