# AIML Module - Focus/Attention Tracking

Real-world AIML module for predicting distraction from **posture detection** and **tab switching**.

## 🚀 Quick Start

### 1. Setup

```bash
cd aiml
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
python setup.py
```

### 2. Run Real-World Demo

**Combined (Posture + Tab Switching):**
```bash
python realtime_demo.py
```

**Posture Detection Only:**
```bash
python realtime_posture_detector.py
```

**Tab Monitoring Only:**
```bash
python tab_switch_monitor.py
```

## 📁 What's Included

- ✅ **Real-time posture detection** from webcam
- ✅ **Tab switching monitoring** 
- ✅ **AIML distraction prediction**
- ✅ **Session focus scoring**
- ✅ **Trained models** (ready to use)

## 📚 Documentation

- **Real-World Usage**: `REAL_WORLD_USAGE.md`
- **Quick Start**: `QUICK_START_AIML.md`
- **Full API Docs**: `README.md` (this file)
- **Setup Guide**: `SETUP_AIML.md`

## 🎯 Features

### Posture Detection
- Real-time webcam analysis
- Spine angle calculation
- Slouch detection
- Looking away detection
- Visual feedback with overlays

### Tab Switching
- Automatic tab switch detection
- Frequency calculation
- Cross-platform (Windows/Linux/Mac)
- Background monitoring

### AIML Prediction
- Real-time distraction probability
- Focus score calculation
- Session-level analysis

## 💻 Usage Examples

### Posture Detection
```python
from realtime_posture_detector import PostureDetector

detector = PostureDetector()
detector.run(user_id=1)
```

### Tab Monitoring
```python
from tab_switch_monitor import TabSwitchMonitor

monitor = TabSwitchMonitor()
monitor.monitor(user_id=1)
```

### Combined
```python
python realtime_demo.py
```

## 📦 Requirements

- Python 3.8+
- Webcam (for posture detection)
- Browser (for tab monitoring)

**Dependencies:**
- pandas, numpy, scikit-learn, xgboost
- opencv-python, mediapipe
- pywin32 (Windows only)

## 🔧 Installation Issues

**Windows:**
```bash
pip install pywin32
```

**Linux:**
```bash
sudo apt-get install xdotool
```

**Mac:**
- Works out of the box

## 📖 See Also

- `example_usage.py` - Basic usage examples
- `REAL_WORLD_USAGE.md` - Real-world usage guide
- `QUICK_START_AIML.md` - Quick start guide
