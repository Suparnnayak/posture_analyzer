# 👁️ User Concentration Monitor API

Flask API that monitors user concentration by tracking:
- **Eye position** - Alerts if looking away from screen for >10 seconds
- **Posture** - Determines user interest level based on spine angle

Returns JSON for frontend integration.

## 🚀 Quick Start

### 1. Setup

```bash
cd aiml
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Run Locally

```bash
cd aiml
python app.py
```

API will be available at `http://localhost:5000`

## 📡 API Endpoints

### `GET /`
Health check and API info.

**Response:**
```json
{
  "status": "ok",
  "message": "User Concentration Monitor API",
  "endpoints": {
    "/api/analyze": "POST - Analyze image (base64 or webcam)",
    "/api/health": "GET - Health check"
  }
}
```

### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "concentration-monitor"
}
```

### `POST /api/analyze`
Analyze frame for eye tracking and posture.

**Request Options:**

1. **With base64 image (recommended for production):**
```json
{
  "image": "base64_encoded_image_string"
}
```

2. **Empty body (uses webcam if available):**
```json
{}
```

**Response:**
```json
{
  "timestamp": 1731652010000,
  "eye_tracking": {
    "looking_away": false,
    "duration": 0.0,
    "confidence": 0.15,
    "head_yaw": 2.5,
    "head_pitch": -1.2,
    "eye_aspect_ratio": 1.45
  },
  "posture": {
    "interest_score": 0.65,
    "interest_level": "high",
    "spine_angle": 12.5,
    "slouch": false,
    "visibility_score": 0.99
  },
  "alert": null
}
```

**Alert Response (when looking away >10s):**
```json
{
  "timestamp": 1731652010000,
  "eye_tracking": {
    "looking_away": true,
    "duration": 12.5,
    "confidence": 0.85
  },
  "posture": {
    "interest_score": 0.45,
    "interest_level": "medium",
    "spine_angle": 22.3,
    "slouch": false
  },
  "alert": {
    "triggered": true,
    "message": "⚠️ ALERT: Looking away for 12.5 seconds!",
    "duration": 12.5,
    "threshold": 10.0
  }
}
```

## 🌐 Frontend Integration

### JavaScript Example

```javascript
// Send base64 image from webcam
async function analyzeFrame(videoElement) {
  // Convert video frame to base64
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0);
  const base64Image = canvas.toDataURL('image/jpeg').split(',')[1];
  
  // Send to API
  const response = await fetch('http://localhost:5000/api/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image: base64Image })
  });
  
  const data = await response.json();
  
  // Handle alert
  if (data.alert?.triggered) {
    showNotification(data.alert.message);
  }
  
  // Update UI
  updateInterestLevel(data.posture.interest_level);
  updateEyeStatus(data.eye_tracking.looking_away);
  
  return data;
}

// Continuous monitoring
setInterval(() => {
  analyzeFrame(videoElement);
}, 100); // ~10 FPS
```

### Python Example

```python
import requests
import base64
import cv2

# Capture frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

# Encode to base64
_, buffer = cv2.imencode('.jpg', frame)
image_base64 = base64.b64encode(buffer).decode('utf-8')

# Send to API
response = requests.post(
    'http://localhost:5000/api/analyze',
    json={'image': image_base64}
)

data = response.json()
print(data)
```

## 🚀 Deploy on Render

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `concentration-monitor`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r aiml/requirements.txt`
   - **Start Command**: `cd aiml && gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120 app:app`
   - **Port**: Auto-detected
5. Click **"Create Web Service"**

### 3. Using render.yaml (Alternative)

If you have `render.yaml` in your repo root, Render will auto-detect it:

1. Push `render.yaml` to your repo
2. Go to Render Dashboard
3. Click **"New +"** → **"Blueprint"**
4. Select your repository
5. Render will use the configuration from `render.yaml`

## 📦 Requirements

- Python 3.8+
- Webcam (for local testing without base64)
- MediaPipe (for eye tracking)
- OpenCV (for image processing)
- Flask (for API)
- Gunicorn (for production)

## 🔧 How It Works

1. **Eye Tracking**: Uses MediaPipe Face Mesh to detect eye position and head pose
2. **Looking Away Detection**: Calculates head yaw/pitch and eye offset from center
3. **Duration Tracking**: Tracks how long user has been looking away
4. **Alert System**: Triggers alert when duration >10 seconds
5. **Posture Analysis**: Uses MediaPipe Pose to analyze spine/neck angle
6. **Interest Calculation**: Based on posture quality

## 🎯 Features

- ✅ RESTful API with JSON responses
- ✅ CORS enabled for frontend integration
- ✅ Base64 image support
- ✅ Webcam fallback (local testing)
- ✅ Real-time eye tracking
- ✅ Posture analysis
- ✅ Alert system (>10s looking away)
- ✅ Production-ready (Gunicorn)

## 📝 Notes

- For production, always send base64 images (webcam not available on Render)
- API maintains state for "looking away" duration tracking
- All responses are JSON formatted
- CORS is enabled for cross-origin requests

---

**Simple, Clean, API-Ready** - Just send images and get concentration metrics!
