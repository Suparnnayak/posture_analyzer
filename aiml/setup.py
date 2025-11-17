"""
Setup script for AIML module.
Installs dependencies and trains initial models.
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"[OK] {description}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed")
        if e.stderr:
            print(e.stderr)
        return False

def main():
    print("=" * 60)
    print("AIML Module Setup")
    print("=" * 60)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check if we're in the right directory
    if not Path("models").exists():
        print("\n[ERROR] Please run this script from the aiml/ directory")
        return
    
    # Install dependencies
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies"
    ):
        print("\n[WARNING] Some dependencies may not have installed correctly")
    
    # Check if data exists
    data_dir = Path("data")
    events_file = data_dir / "events.csv"
    sessions_file = data_dir / "sessions.csv"
    
    if not events_file.exists():
        print(f"\n[WARNING] {events_file} not found")
        print("Models cannot be trained without training data")
        return
    
    if not sessions_file.exists():
        print(f"\n[WARNING] {sessions_file} not found")
        print("Session model cannot be trained without training data")
        return
    
    # Train models
    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)
    
    run_command(
        f"{sys.executable} models/train_behavior_model.py",
        "Training behavior model"
    )
    
    run_command(
        f"{sys.executable} models/train_session_model.py",
        "Training session model"
    )
    
    # Verify models exist
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    behavior_model = Path("models/behavior_model.pkl")
    session_model = Path("models/session_model.pkl")
    
    if behavior_model.exists():
        print(f"[OK] Behavior model: {behavior_model}")
    else:
        print(f"[ERROR] Behavior model not found: {behavior_model}")
    
    if session_model.exists():
        print(f"[OK] Session model: {session_model}")
    else:
        print(f"[ERROR] Session model not found: {session_model}")
    
    # Test imports
    print("\nTesting imports...")
    try:
        from inference.predictor import FocusPredictor
        from inference.session_scorer import SessionScorer
        print("[OK] Imports successful")
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test prediction: python -c \"from inference.predictor import FocusPredictor; p = FocusPredictor(); print('OK')\"")
    print("2. Integrate into your backend")
    print("3. See README.md for usage examples")

if __name__ == "__main__":
    main()

