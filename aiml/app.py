"""
Flask API for User Concentration Monitoring

Tracks:
1. Eye position - alerts if looking away >10 seconds
2. Posture - determines user interest level

Accepts base64 image or uses webcam, returns JSON.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import time
import base64
import threading
from eye_tracker import EyeTracker
from posture_analyzer import PostureAnalyzer

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize trackers (singleton pattern)
eye_tracker = None
posture_analyzer = None
looking_away_state = {'start_time': None}
alert_threshold = 10.0  # 10 seconds

def init_trackers():
    """Initialize eye tracker and posture analyzer."""
    global eye_tracker, posture_analyzer
    if eye_tracker is None:
        eye_tracker = EyeTracker()
    if posture_analyzer is None:
        posture_analyzer = PostureAnalyzer()
    return eye_tracker, posture_analyzer

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None

def process_frame(frame):
    """Process a single frame and return metrics."""
    global looking_away_state
    
    # Initialize trackers if needed
    eye_tracker, posture_analyzer = init_trackers()
    
    current_time = time.time()
    
    # Process eye tracking
    eye_result = eye_tracker.process_frame(frame)
    
    # Process posture
    posture_result = posture_analyzer.process_frame(frame)
    
    # Track looking away duration
    looking_away = False
    looking_away_duration = 0.0
    alert_triggered = False
    
    if eye_result:
        looking_away = bool(eye_result["looking_away"])
        
        if looking_away:
            if looking_away_state.get('start_time') is None:
                looking_away_state['start_time'] = current_time
            
            looking_away_duration = current_time - looking_away_state['start_time']
            
            if looking_away_duration >= alert_threshold:
                alert_triggered = True
        else:
            if looking_away_state.get('start_time') is not None:
                looking_away_state['start_time'] = None
    
    # Build metrics
    metrics = {
        "timestamp": int(current_time * 1000),
        "eye_tracking": {
            "looking_away": bool(looking_away),
            "duration": float(round(looking_away_duration, 2)),
            "confidence": float(eye_result.get("confidence", 0.0)) if eye_result else 0.0,
            "head_yaw": float(eye_result.get("head_yaw", 0.0)) if eye_result else 0.0,
            "head_pitch": float(eye_result.get("head_pitch", 0.0)) if eye_result else 0.0,
            "eye_aspect_ratio": float(eye_result.get("eye_aspect_ratio", 0.0)) if eye_result else 0.0
        } if eye_result else None,
        "posture": {
            "interest_score": float(posture_result["interest_score"]),
            "interest_level": str(posture_result["interest_level"]),
            "spine_angle": float(posture_result["spine_angle"]),
            "slouch": bool(posture_result["slouch"]),
            "visibility_score": float(posture_result.get("visibility_score", 1.0))
        } if posture_result else None,
        "alert": {
            "triggered": bool(alert_triggered),
            "message": f"⚠️ ALERT: Looking away for {looking_away_duration:.1f} seconds!" if alert_triggered else None,
            "duration": float(round(looking_away_duration, 2)),
            "threshold": float(alert_threshold)
        } if alert_triggered else None
    }
    
    return metrics

@app.route('/')
def index():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "message": "User Concentration Monitor API",
        "endpoints": {
            "/api/analyze": "POST - Analyze image (base64 or webcam)",
            "/api/health": "GET - Health check"
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "concentration-monitor"})

@app.route('/api/analyze', methods=['GET', 'POST'])
def analyze():
    """
    Analyze frame for eye tracking and posture.
    
    GET: Returns API usage information
    POST: Analyzes image
        - JSON with 'image' (base64 string) OR
        - Empty body to use webcam (if available)
    
    Returns:
    - JSON with eye tracking, posture, and alert data
    """
    # Handle GET request (for browser testing)
    if request.method == 'GET':
        return jsonify({
            "message": "Use POST method to analyze images",
            "usage": {
                "method": "POST",
                "content_type": "application/json",
                "body": {
                    "image": "base64_encoded_image_string (optional - uses webcam if not provided)"
                },
                "example": {
                    "image": "iVBORw0KGgoAAAANS..."
                }
            },
            "curl_example": "curl -X POST http://127.0.0.1:5000/api/analyze -H 'Content-Type: application/json' -d '{\"image\": \"base64_string\"}'"
        }), 200
    try:
        frame = None
        
        # Check if image is provided in request
        if request.is_json:
            data = request.get_json()
            image_base64 = data.get('image', '')
            
            if image_base64:
                # Convert base64 to image
                frame = base64_to_image(image_base64)
                if frame is None:
                    return jsonify({
                        "error": "Invalid base64 image",
                        "status": "error"
                    }), 400
            else:
                # Try to use webcam
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    return jsonify({
                        "error": "No image provided and webcam not available",
                        "status": "error",
                        "hint": "Send base64 image in JSON: {'image': 'base64_string'}"
                    }), 400
                
                ret, frame = cap.read()
                cap.release()
                
                if not ret or frame is None:
                    return jsonify({
                        "error": "Could not capture frame from webcam",
                        "status": "error"
                    }), 400
        else:
            # Try to use webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({
                    "error": "No image provided and webcam not available",
                    "status": "error",
                    "hint": "Send base64 image in JSON: {'image': 'base64_string'}"
                }), 400
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return jsonify({
                    "error": "Could not capture frame from webcam",
                    "status": "error"
                }), 400
        
        # Process frame
        metrics = process_frame(frame)
        
        return jsonify(metrics), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == '__main__':
    # Initialize trackers on startup
    init_trackers()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

