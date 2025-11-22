# 🚀 Deployment Guide

## Local Testing

### 1. Setup Environment

```bash
cd aiml
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Run Flask App

```bash
cd aiml
python app.py
```

API will be available at `http://localhost:5000`

### 3. Test API

**Health Check:**
```bash
curl http://localhost:5000/api/health
```

**Analyze with Webcam (if available):**
```bash
curl -X POST http://localhost:5000/api/analyze
```

**Analyze with Base64 Image:**
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_string"}'
```

## Deploy on Render

### Option 1: Using Render Dashboard

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Create Web Service on Render:**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click **"New +"** → **"Web Service"**
   - Connect your GitHub repository
   - Configure:
     - **Name**: `concentration-monitor`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r aiml/requirements.txt`
     - **Start Command**: `cd aiml && gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120 app:app`
     - **Port**: Auto-detected (leave empty)
   - Click **"Create Web Service"**

3. **Wait for Deployment:**
   - Render will build and deploy automatically
   - Your API will be available at `https://your-app-name.onrender.com`

### Option 2: Using render.yaml (Recommended)

1. **Push to GitHub** (same as above)

2. **Create Blueprint on Render:**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click **"New +"** → **"Blueprint"**
   - Select your repository
   - Render will automatically detect `render.yaml` and configure the service

3. **Deploy:**
   - Click **"Apply"**
   - Render will deploy using the configuration from `render.yaml`

## Environment Variables

No environment variables required. The app works out of the box.

## Production Notes

- **Webcam**: Not available on Render. Always send base64 images in production.
- **Workers**: Configured for 2 workers in `render.yaml` (adjust based on your needs)
- **Timeout**: Set to 120 seconds for image processing
- **CORS**: Enabled for frontend integration

## Testing Deployed API

```bash
# Health check
curl https://your-app-name.onrender.com/api/health

# Analyze with base64 image
curl -X POST https://your-app-name.onrender.com/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_string"}'
```

## Frontend Integration

Your frontend can call the API like this:

```javascript
const response = await fetch('https://your-app-name.onrender.com/api/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ image: base64Image })
});

const data = await response.json();
```

## Troubleshooting

### Build Fails
- Check that `aiml/requirements.txt` exists
- Verify Python version (3.8+)

### API Returns 500 Error
- Check Render logs for detailed error messages
- Ensure base64 image is valid
- Verify MediaPipe and OpenCV are installed correctly

### CORS Issues
- CORS is enabled by default
- If issues persist, check frontend request headers

---

**Ready to deploy!** 🎉

