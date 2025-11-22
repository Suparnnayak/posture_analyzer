# 🚀 Quick Start Guide

## Local Testing

### 1. Install Dependencies

```bash
cd aiml
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Run Flask API

```bash
cd aiml
python app.py
```

API will start at `http://localhost:5000`

### 3. Test API

**Health Check:**
```bash
curl http://localhost:5000/api/health
```

**Analyze Frame (with webcam):**
```bash
curl -X POST http://localhost:5000/api/analyze
```

**Analyze Frame (with base64 image):**
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_string"}'
```

## Deploy on Render

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `concentration-monitor`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r aiml/requirements.txt`
   - **Start Command**: `cd aiml && gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120 app:app`
5. Click **"Create Web Service"**

### Step 3: Test Deployed API

```bash
curl https://your-app-name.onrender.com/api/health
```

## Frontend Integration

```javascript
// Send base64 image from webcam
const canvas = document.createElement('canvas');
canvas.width = videoElement.videoWidth;
canvas.height = videoElement.videoHeight;
const ctx = canvas.getContext('2d');
ctx.drawImage(videoElement, 0, 0);
const base64Image = canvas.toDataURL('image/jpeg').split(',')[1];

// Call API
const response = await fetch('https://your-api-url/api/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ image: base64Image })
});

const data = await response.json();
console.log(data);
```

---

**That's it!** Your API is ready to use! 🎉

