"""
Feature Engineering for Posture and Tab Switching Events

This module processes raw events and creates features for ML models.
Supports both posture detection and tab switching events.
"""
import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame):
    """
    Engineer features from raw events for distraction prediction.
    
    Expected input columns:
    - timestamp: milliseconds since epoch
    - spineAngle: angle in degrees (0-90) for posture events
    - slouch: 0 or 1 for posture events
    - lookingAway: 0 or 1 for posture events
    - eventType: "posture", "tab_switch", "break_start", "break_end", "looking_away"
    - duration: duration in milliseconds
    - user_id: user identifier (optional)
    - tabCount: number of tabs switched (for tab_switch events, optional)
    - tabFrequency: frequency of tab switches (optional)
    
    Returns:
        tuple: (feature_df, labels) where feature_df contains engineered features
               and labels is the target variable (1 = distracted, 0 = focused)
    """
    df = df.copy()
    df = df.fillna(0)
    
    # -------------------------
    # BASIC FEATURES
    # -------------------------
    # Posture features
    df["is_slouching"] = df["slouch"].astype(int) if "slouch" in df.columns else 0
    df["is_looking_away"] = df["lookingAway"].astype(int) if "lookingAway" in df.columns else 0
    
    # Normalize spine angle (0-90 degrees -> 0-1)
    if "spineAngle" in df.columns:
        df["norm_angle"] = df["spineAngle"].clip(0, 90) / 90
    else:
        df["norm_angle"] = 0.0
    
    # Normalize duration (0-300 seconds -> 0-1)
    df["norm_break_duration"] = df["duration"].clip(0, 300000) / 300000
    
    # Tab switching features
    if "tabCount" in df.columns:
        df["tab_count"] = df["tabCount"].astype(int)
    else:
        df["tab_count"] = 0
    
    if "tabFrequency" in df.columns:
        df["tab_frequency"] = df["tabFrequency"].astype(float)
    else:
        df["tab_frequency"] = 0.0
    
    # Label: 1 = distracted
    distracted_events = ["break_start", "looking_away", "tab_switch"]
    df["label"] = df["eventType"].apply(
        lambda x: 1 if x in distracted_events else 0
    )
    
    # -------------------------
    # TIME FEATURES
    # -------------------------
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    
    # Extract time features
    df["hour"] = df["timestamp"].dt.hour.fillna(12)  # Default to noon if invalid
    df["minute"] = df["timestamp"].dt.minute.fillna(0)
    df["dayofweek"] = df["timestamp"].dt.dayofweek.fillna(0)
    
    # Normalize time of day using sin/cos (circular encoding)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # -------------------------
    # SESSION FEATURES (Rolling windows)
    # -------------------------
    df = df.sort_values("timestamp")
    
    # Event index
    df["event_index"] = range(len(df))
    
    # Rolling averages for posture
    df["slouch_freq"] = df["is_slouching"].rolling(10, min_periods=1).mean()
    df["lookaway_freq"] = df["is_looking_away"].rolling(10, min_periods=1).mean()
    
    # Break frequency
    df["break_freq"] = (df["eventType"] == "break_start").rolling(20, min_periods=1).mean()
    
    # Tab switching frequency (rolling window)
    df["tab_switch_freq"] = (df["eventType"] == "tab_switch").rolling(20, min_periods=1).mean()
    
    # Cumulative tab switches in session
    df["cumulative_tabs"] = (df["eventType"] == "tab_switch").cumsum()
    
    # Cumulative time since session start
    valid_timestamps = df["timestamp"].notna()
    if valid_timestamps.any():
        start_time = df.loc[valid_timestamps, "timestamp"].iloc[0]
        df["session_elapsed"] = (df["timestamp"] - start_time).dt.total_seconds()
        df["session_elapsed"] = df["session_elapsed"].fillna(0)
    else:
        df["session_elapsed"] = 0
    
    # -------------------------
    # USER-LEVEL FEATURES
    # -------------------------
    if "user_id" in df.columns and df["user_id"].notna().any():
        # Per-user baselines
        df["user_slouch_baseline"] = df.groupby("user_id")["is_slouching"].transform("mean")
        df["user_break_baseline"] = df.groupby("user_id")["break_freq"].transform("mean")
        df["user_tab_baseline"] = df.groupby("user_id")["tab_switch_freq"].transform("mean")
    else:
        df["user_slouch_baseline"] = 0.5
        df["user_break_baseline"] = 0.5
        df["user_tab_baseline"] = 0.5
    
    # -------------------------
    # INTERACTION FEATURES
    # -------------------------
    # Combined distraction indicators
    df["combined_distraction"] = (
        df["is_slouching"] + 
        df["is_looking_away"] + 
        (df["eventType"] == "tab_switch").astype(int)
    ).clip(0, 1)
    
    # -------------------------
    # Final feature list
    # -------------------------
    feature_cols = [
        # Posture features
        "norm_angle",
        "is_slouching",
        "is_looking_away",
        
        # Tab switching features
        "tab_count",
        "tab_frequency",
        "tab_switch_freq",
        "cumulative_tabs",
        
        # Time features
        "norm_break_duration",
        "hour_sin",
        "hour_cos",
        "dayofweek",
        
        # Session features
        "slouch_freq",
        "lookaway_freq",
        "break_freq",
        "session_elapsed",
        
        # User baselines
        "user_slouch_baseline",
        "user_break_baseline",
        "user_tab_baseline",
        
        # Combined features
        "combined_distraction"
    ]
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    return df[feature_cols], df["label"]


def prepare_single_event(event_dict: dict) -> pd.DataFrame:
    """
    Prepare a single event dictionary for feature engineering.
    
    Args:
        event_dict: Dictionary with event data
        
    Returns:
        DataFrame with single row ready for feature engineering
    """
    # Convert to DataFrame
    df = pd.DataFrame([event_dict])
    
    # Ensure required columns exist
    required_cols = ["timestamp", "eventType", "duration"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Set defaults for optional columns
    optional_defaults = {
        "spineAngle": 0.0,
        "slouch": 0,
        "lookingAway": 0,
        "user_id": 1,
        "tabCount": 0,
        "tabFrequency": 0.0
    }
    
    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default
    
    return df

