"""
Interactive Streamlit Dashboard for Real-Time Focus Tracking

Features:
- Real-time posture detection from webcam
- Tab switching monitoring
- Focus alerts (looking away >10s, any tab switch)
- Visual metrics and charts
"""
import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime, timedelta
import threading
import queue
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from realtime_posture_detector import PostureDetector
from tab_switch_monitor import TabSwitchMonitor
from inference.predictor import FocusPredictor
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Focus Tracking Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disable Streamlit's default behavior that might trigger blockers
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Prevent resource blocking issues */
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'posture_data' not in st.session_state:
    st.session_state.posture_data = []
if 'tab_switches' not in st.session_state:
    st.session_state.tab_switches = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'looking_away_start' not in st.session_state:
    st.session_state.looking_away_start = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=2)
if 'metrics_queue' not in st.session_state:
    st.session_state.metrics_queue = queue.Queue()
# Use threading.Event instead of accessing session_state from threads
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()

class EnhancedPostureDetector(PostureDetector):
    """Enhanced posture detector with looking away duration tracking and prediction smoothing."""
    
    def __init__(self):
        super().__init__()
        self.looking_away_start_time = None
        self.lookaway_threshold_seconds = 10.0
        
        # Prediction smoothing for better accuracy
        self.distraction_history = []  # Store recent predictions
        self.smoothing_window = 5  # Number of predictions to average
        self.smoothed_distraction_prob = 0.0
        self.alpha = 0.3  # Exponential moving average factor (0-1, lower = more smoothing)
        
        # Calibration thresholds
        self.distraction_threshold = 0.6  # Higher threshold to reduce false positives
        self.confidence_threshold = 0.15  # Minimum change to consider significant
    
    def check_looking_away_duration(self, is_looking_away):
        """
        Check if user has been looking away for more than threshold.
        
        Returns:
            tuple: (is_looking_away_now, has_been_away_too_long, duration)
        """
        current_time = time.time()
        
        if is_looking_away:
            if self.looking_away_start_time is None:
                self.looking_away_start_time = current_time
            
            duration = current_time - self.looking_away_start_time
            too_long = duration >= self.lookaway_threshold_seconds
            
            return True, too_long, duration
        else:
            # Reset if not looking away
            if self.looking_away_start_time is not None:
                self.looking_away_start_time = None
            return False, False, 0.0
    
    def smooth_prediction(self, raw_prob):
        """
        Apply exponential moving average smoothing to reduce noise.
        
        Args:
            raw_prob: Raw distraction probability from model
            
        Returns:
            Smoothed probability (0-1)
        """
        # Exponential moving average
        if self.smoothed_distraction_prob == 0.0:
            # First prediction - use raw value
            self.smoothed_distraction_prob = raw_prob
        else:
            # EMA: new = alpha * raw + (1 - alpha) * old
            self.smoothed_distraction_prob = (
                self.alpha * raw_prob + 
                (1 - self.alpha) * self.smoothed_distraction_prob
            )
        
        # Also maintain a rolling window for additional stability
        self.distraction_history.append(raw_prob)
        if len(self.distraction_history) > self.smoothing_window:
            self.distraction_history.pop(0)
        
        # Use average of EMA and window average for final value
        window_avg = sum(self.distraction_history) / len(self.distraction_history)
        final_prob = 0.7 * self.smoothed_distraction_prob + 0.3 * window_avg
        
        # Clamp to valid range
        return max(0.0, min(1.0, final_prob))
    
    def calibrate_prediction(self, smoothed_prob, spine_angle, slouch, looking_away):
        """
        Apply calibration based on physical indicators.
        
        Args:
            smoothed_prob: Smoothed distraction probability
            spine_angle: Current spine angle
            slouch: Whether slouching (0 or 1)
            looking_away: Whether looking away (0 or 1)
            
        Returns:
            Calibrated probability
        """
        # Base probability from model
        calibrated = smoothed_prob
        
        # Adjust based on physical indicators
        if looking_away:
            # If actually looking away, increase probability
            calibrated = min(1.0, calibrated + 0.2)
        elif not looking_away and smoothed_prob > 0.7:
            # If model says distracted but not looking away, reduce
            calibrated = max(0.0, calibrated - 0.15)
        
        if slouch:
            # Slouching increases distraction probability
            calibrated = min(1.0, calibrated + 0.1)
        
        # Spine angle adjustment (very upright = less distracted)
        if spine_angle < 10:  # Very good posture
            calibrated = max(0.0, calibrated - 0.1)
        elif spine_angle > 30:  # Poor posture
            calibrated = min(1.0, calibrated + 0.15)
        
        return calibrated

class EnhancedTabMonitor(TabSwitchMonitor):
    """Enhanced tab monitor with immediate alerts."""
    
    def __init__(self):
        super().__init__()
        self.alert_callback = None
    
    def set_alert_callback(self, callback):
        """Set callback function for tab switch alerts."""
        self.alert_callback = callback
    
    def create_tab_event(self, user_id=1):
        """Create tab switch event and trigger alert."""
        event = super().create_tab_event(user_id)
        
        # Trigger immediate alert
        if self.alert_callback:
            self.alert_callback({
                "type": "tab_switch",
                "message": f"Tab switched! (Total: {event['tabCount']})",
                "timestamp": datetime.now(),
                "severity": "warning"
            })
        
        return event

def run_posture_detection(detector, user_id, frame_queue, metrics_queue, stop_event):
    """Run posture detection in background thread."""
    # Try multiple camera indices
    cap = None
    camera_index = 0
    
    # Try to find working camera
    for i in range(3):  # Try cameras 0, 1, 2
        test_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows for better compatibility
        if test_cap.isOpened():
            # Give camera time to initialize
            time.sleep(0.5)
            ret, test_frame = test_cap.read()
            if ret and test_frame is not None:
                cap = test_cap
                camera_index = i
                metrics_queue.put({
                    "type": "info",
                    "message": f"Camera {i} opened successfully"
                })
                break
            else:
                test_cap.release()
        else:
            test_cap.release()
    
    if cap is None or not cap.isOpened():
        error_msg = (
            "Could not open webcam. Possible causes:\n"
            "1. Camera is being used by another app\n"
            "2. Camera permissions not granted\n"
            "3. No camera detected\n"
            "4. Try closing other apps using the camera"
        )
        metrics_queue.put({"error": error_msg})
        return
    
    # Set camera properties for better performance
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except:
        pass  # Some cameras don't support all properties
    
    # CRITICAL: Warm up camera by reading multiple frames
    # This prevents the "flash then turn off" issue
    metrics_queue.put({
        "type": "info",
        "message": "Warming up camera..."
    })
    for warmup in range(15):  # Read 15 frames to stabilize
        ret, _ = cap.read()
        if ret:
            time.sleep(0.05)  # Small delay between warmup frames
    metrics_queue.put({
        "type": "info",
        "message": "Camera ready!"
    })
    
    last_prediction_time = 0
    prediction_interval = 0.5
    consecutive_failures = 0
    
    try:
        # Use threading.Event instead of accessing session_state (avoids ScriptRunContext warnings)
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 20:
                    metrics_queue.put({
                        "type": "warning",
                        "message": "Camera read failures. Check connection."
                    })
                    time.sleep(0.5)
                continue
            
            consecutive_failures = 0  # Reset on successful read
            
            # Process frame
            result = detector.process_frame(frame, user_id)
            
            if result:
                event, landmarks = result
                spine_angle = event["spineAngle"]
                slouch = event["slouch"]
                looking_away = event["lookingAway"]
                
                # Check looking away duration
                is_away_now, too_long, duration = detector.check_looking_away_duration(looking_away)
                
                # Draw pose on frame
                frame = detector.draw_pose(frame, landmarks, spine_angle, slouch, looking_away)
                
                # Add looking away duration info
                if is_away_now:
                    duration_text = f"Looking Away: {duration:.1f}s"
                    if too_long:
                        cv2.putText(frame, "⚠ NOT FOCUSED!", (10, 210),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        duration_text += " (ALERT!)"
                    cv2.putText(frame, duration_text, (10, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Predict distraction (with smoothing and calibration)
                current_time = time.time()
                if detector.predictor and (current_time - last_prediction_time) >= prediction_interval:
                    # Get raw prediction from model
                    raw_prediction = detector.predictor.predict(event)
                    raw_prob = raw_prediction['distraction_prob']
                    
                    # Apply smoothing to reduce noise
                    smoothed_prob = detector.smooth_prediction(raw_prob)
                    
                    # Apply calibration based on physical indicators
                    calibrated_prob = detector.calibrate_prediction(
                        smoothed_prob, spine_angle, slouch, looking_away
                    )
                    
                    # Use calibrated threshold for binary decision
                    is_distracted = calibrated_prob >= detector.distraction_threshold
                    
                    # Update event with improved predictions
                    event['distraction_prob'] = calibrated_prob
                    event['focus_prob'] = 1.0 - calibrated_prob
                    event['is_distracted'] = is_distracted
                    event['raw_distraction_prob'] = raw_prob  # Keep raw for debugging
                    event['smoothed_distraction_prob'] = smoothed_prob
                    
                    # Check if should alert (looking away too long)
                    if too_long:
                        event['alert'] = True
                        event['alert_reason'] = 'looking_away_too_long'
                        metrics_queue.put({
                            "type": "alert",
                            "message": f"Looking away for {duration:.1f} seconds!",
                            "timestamp": datetime.now(),
                            "severity": "error"
                        })
                    
                    last_prediction_time = current_time
                
                # Convert frame for Streamlit (BGR to RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Put frame in queue (non-blocking, drop old frames if full)
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()  # Remove oldest frame
                    except:
                        pass
                try:
                    frame_queue.put_nowait(frame_rgb)
                except:
                    pass  # Queue full, skip this frame
                
                # Put metrics in queue (only if we have a prediction or posture data)
                metrics = {
                    "timestamp": datetime.now(),
                    "spine_angle": spine_angle,
                    "slouch": slouch,
                    "looking_away": looking_away,
                    "looking_away_duration": duration if is_away_now else 0.0,
                    "too_long": too_long,
                    "distraction_prob": event.get('distraction_prob', 0.0),
                    "focus_prob": event.get('focus_prob', 1.0),
                    "is_distracted": event.get('is_distracted', False)
                }
                
                # Only add raw/smoothed if available (for debugging)
                if 'raw_distraction_prob' in event:
                    metrics['raw_distraction_prob'] = event['raw_distraction_prob']
                    metrics['smoothed_distraction_prob'] = event.get('smoothed_distraction_prob', 0.0)
                
                metrics_queue.put(metrics)
    
    except Exception as e:
        metrics_queue.put({"error": str(e)})
    finally:
        cap.release()

def run_tab_monitoring(monitor, user_id, metrics_queue, stop_event):
    """Run tab monitoring in background thread."""
    # Set alert callback
    def alert_callback(alert_data):
        metrics_queue.put(alert_data)
    
    monitor.set_alert_callback(alert_callback)
    
    # Check if tab monitoring is supported
    import platform
    system = platform.system()
    
    # Check dependencies
    can_monitor = False
    missing_deps = []
    
    if system == "Windows":
        try:
            import win32gui
            can_monitor = True
        except ImportError:
            missing_deps.append("pywin32 (pip install pywin32)")
    elif system == "Linux":
        import subprocess
        try:
            subprocess.run(['xdotool', '--version'], capture_output=True, check=True)
            can_monitor = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_deps.append("xdotool (sudo apt-get install xdotool)")
    elif system == "Darwin":  # Mac
        can_monitor = True  # Mac should work with osascript
    
    if not can_monitor:
        error_msg = f"Tab monitoring not available. Install: {', '.join(missing_deps)}"
        metrics_queue.put({
            "type": "warning",
            "message": error_msg,
            "timestamp": datetime.now()
        })
        # Continue anyway, just won't detect tab switches
    
    previous_title = None
    check_interval = 0.5
    
    try:
        # Use threading.Event instead of accessing session_state (avoids ScriptRunContext warnings)
        while not stop_event.is_set():
            try:
                if can_monitor:
                    current_title = monitor.get_active_window_title()
                    
                    if monitor.detect_tab_switch(current_title, previous_title):
                        event = monitor.create_tab_event(user_id)
                        
                        # Predict distraction
                        if monitor.predictor:
                            prediction = monitor.predictor.predict(event)
                            metrics_queue.put({
                                "type": "tab_switch",
                                "timestamp": datetime.now(),
                                "tab_count": event['tabCount'],
                                "tab_frequency": event['tabFrequency'],
                                "distraction_prob": prediction['distraction_prob'],
                                "is_distracted": prediction['is_distracted']
                            })
                    
                    previous_title = current_title
            except Exception as e:
                # Log error but continue
                error_str = str(e)
                if "win32gui" in error_str or "xdotool" in error_str:
                    # Dependency error - already handled above
                    pass
                else:
                    # Other error - log it
                    metrics_queue.put({
                        "type": "warning",
                        "message": f"Tab monitoring: {error_str}",
                        "timestamp": datetime.now()
                    })
            
            time.sleep(check_interval)
    
    except Exception as e:
        metrics_queue.put({"error": f"Tab monitoring error: {str(e)}"})

def main():
    """Main dashboard function."""
    st.title("🎯 Real-Time Focus Tracking Dashboard")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        user_id = st.number_input("User ID", min_value=1, value=1, step=1)
        
        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("▶️ Start", type="primary", use_container_width=True)
        with col2:
            stop_btn = st.button("⏹️ Stop", use_container_width=True)
        
        if start_btn:
            # Reset thread state to allow restart
            if 'threads_started' in st.session_state:
                del st.session_state.threads_started
            st.session_state.is_running = True
            st.session_state.stop_event.clear()  # Clear stop event to allow threads to run
            st.rerun()
        
        if stop_btn:
            st.session_state.is_running = False
            st.session_state.stop_event.set()  # Signal threads to stop
            # Clear thread state
            if 'threads_started' in st.session_state:
                del st.session_state.threads_started
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Settings")
        lookaway_threshold = st.slider(
            "Looking Away Alert (seconds)",
            min_value=5,
            max_value=30,
            value=10,
            step=1
        )
        
        distraction_threshold = st.slider(
            "Distraction Threshold",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Higher = fewer false positives, Lower = more sensitive"
        )
        
        smoothing_factor = st.slider(
            "Smoothing Factor",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Lower = more smoothing (less noise), Higher = more responsive"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ Info")
        st.info("""
        **Alerts:**
        - ⚠️ Tab switch: Immediate alert
        - ⚠️ Looking away >10s: Focus alert
        
        **Press 'q' in video to stop**
        """)
    
    # Main content
    if not st.session_state.is_running:
        st.info("👆 Click 'Start' in the sidebar to begin monitoring")
        
        # Diagnostic information
        with st.expander("🔍 System Diagnostics", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Camera Status")
                # Test camera access
                test_cap = cv2.VideoCapture(0)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        st.success("✅ Camera is accessible")
                        st.caption(f"Resolution: {int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                    else:
                        st.error("❌ Camera opened but cannot read frames")
                    test_cap.release()
                else:
                    st.error("❌ Cannot open camera")
                    st.caption("Try: 1) Close other apps using camera 2) Check permissions 3) Restart computer")
            
            with col2:
                st.subheader("Tab Monitoring Status")
                import platform
                system = platform.system()
                st.caption(f"OS: {system}")
                
                if system == "Windows":
                    try:
                        import win32gui
                        st.success("✅ pywin32 installed")
                    except ImportError:
                        st.error("❌ pywin32 not installed")
                        st.code("pip install pywin32", language="bash")
                elif system == "Linux":
                    import subprocess
                    try:
                        subprocess.run(['xdotool', '--version'], capture_output=True, check=True)
                        st.success("✅ xdotool installed")
                    except:
                        st.error("❌ xdotool not installed")
                        st.code("sudo apt-get install xdotool", language="bash")
                else:
                    st.info("✅ Mac - should work")
        
        # Show recent data if available
        if st.session_state.posture_data:
            st.subheader("📈 Recent Data")
            df = pd.DataFrame(st.session_state.posture_data[-50:])
            st.line_chart(df[['spine_angle', 'distraction_prob']])
    else:
        # Initialize detectors
        if 'detector' not in st.session_state:
            st.session_state.detector = EnhancedPostureDetector()
        
        # Update detector settings from sidebar
        st.session_state.detector.lookaway_threshold_seconds = lookaway_threshold
        st.session_state.detector.distraction_threshold = distraction_threshold
        st.session_state.detector.alpha = smoothing_factor
        
        if 'tab_monitor' not in st.session_state:
            st.session_state.tab_monitor = EnhancedTabMonitor()
        
        # Start threads if not already running
        if 'threads_started' not in st.session_state:
            st.session_state.threads_started = True
            
            # Use threading.Event for thread-safe communication
            # This avoids "missing ScriptRunContext" warnings from accessing session_state in threads
            stop_event = st.session_state.stop_event
            
            # Start posture detection thread (no blocking sleep in main thread)
            posture_thread = threading.Thread(
                target=run_posture_detection,
                args=(
                    st.session_state.detector,
                    user_id,
                    st.session_state.frame_queue,
                    st.session_state.metrics_queue,
                    stop_event
                ),
                daemon=True
            )
            posture_thread.start()
            
            # Start tab monitoring thread
            tab_thread = threading.Thread(
                target=run_tab_monitoring,
                args=(
                    st.session_state.tab_monitor,
                    user_id,
                    st.session_state.metrics_queue,
                    stop_event
                ),
                daemon=True
            )
            tab_thread.start()
        
        # Main layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Live Video Feed")
            video_placeholder = st.empty()
            
            # Get latest frame (non-blocking, get most recent)
            frame_displayed = False
            latest_frame = None
            
            # Get the most recent frame, drop older ones
            try:
                while not st.session_state.frame_queue.empty():
                    try:
                        latest_frame = st.session_state.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                if latest_frame is not None:
                    video_placeholder.image(latest_frame, channels="RGB", use_container_width=True)
                    frame_displayed = True
            except Exception as e:
                # Silently handle frame display errors
                pass
            
            if not frame_displayed:
                # Check if there are errors
                error_found = False
                temp_queue = queue.Queue()
                while not st.session_state.metrics_queue.empty():
                    try:
                        item = st.session_state.metrics_queue.get_nowait()
                        if "error" in item:
                            error_found = True
                            video_placeholder.error(f"❌ {item['error']}")
                            if "camera" in item['error'].lower() or "webcam" in item['error'].lower():
                                with st.expander("💡 Fix Camera Issues"):
                                    st.markdown("""
                                    **Quick Fixes:**
                                    1. Close other apps using camera (Zoom, Teams, etc.)
                                    2. Check Windows Settings → Privacy → Camera → Allow desktop apps
                                    3. Restart your computer
                                    4. Run: `python test_camera.py` to test camera
                                    """)
                        else:
                            temp_queue.put(item)
                    except:
                        break
                
                # Put non-error items back
                while not temp_queue.empty():
                    st.session_state.metrics_queue.put(temp_queue.get())
                
                if not error_found:
                    video_placeholder.info("⏳ Waiting for video feed... (this may take a few seconds)")
        
        with col2:
            st.subheader("📊 Current Metrics")
            metrics_placeholder = st.empty()
            
            # Process metrics queue
            metrics_data = {}
            while not st.session_state.metrics_queue.empty():
                try:
                    metric = st.session_state.metrics_queue.get_nowait()
                    
                    if metric.get("type") == "alert":
                        # Add to alerts
                        st.session_state.alerts.append(metric)
                        st.error(f"⚠️ {metric['message']}")
                    elif metric.get("type") == "tab_switch":
                        # Add to tab switches
                        st.session_state.tab_switches.append(metric)
                        st.warning(f"🔄 Tab Switch #{metric.get('tab_count', 0)}")
                    elif metric.get("type") == "warning":
                        # Show warning but don't add to alerts
                        st.warning(f"⚠️ {metric.get('message', 'Warning')}")
                    elif metric.get("type") == "info":
                        # Info message
                        st.info(f"ℹ️ {metric.get('message', 'Info')}")
                    elif "error" in metric:
                        st.error(f"❌ {metric['error']}")
                        # Show detailed error help
                        if "webcam" in metric['error'].lower() or "camera" in metric['error'].lower():
                            with st.expander("💡 How to fix camera issues"):
                                st.markdown("""
                                1. **Close other apps** using the camera (Zoom, Teams, etc.)
                                2. **Check Windows Privacy Settings**:
                                   - Settings → Privacy → Camera
                                   - Allow apps to access camera
                                3. **Restart your computer**
                                4. **Try different camera**: Edit dashboard.py line 127, change `0` to `1`
                                5. **Check if camera works** in other apps first
                                """)
                    else:
                        # Posture metrics
                        metrics_data = metric
                        st.session_state.posture_data.append(metric)
                except queue.Empty:
                    break
            
            # Display current metrics
            if metrics_data:
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Spine Angle", f"{metrics_data.get('spine_angle', 0):.1f}°")
                with m2:
                    st.metric("Distraction", f"{metrics_data.get('distraction_prob', 0)*100:.1f}%")
                with m3:
                    st.metric("Focus", f"{metrics_data.get('focus_prob', 1)*100:.1f}%")
                
                # Looking away duration
                if metrics_data.get('looking_away_duration', 0) > 0:
                    duration = metrics_data['looking_away_duration']
                    if metrics_data.get('too_long', False):
                        st.error(f"⚠️ Looking Away: {duration:.1f}s (NOT FOCUSED!)")
                    else:
                        st.warning(f"👀 Looking Away: {duration:.1f}s")
        
        # Charts section
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("📈 Posture Trends")
            if st.session_state.posture_data:
                df = pd.DataFrame(st.session_state.posture_data[-100:])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['spine_angle'],
                    name='Spine Angle',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['distraction_prob'] * 100,
                    name='Distraction %',
                    line=dict(color='red'),
                    yaxis='y2'
                ))
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Spine Angle (°)",
                    yaxis2=dict(title="Distraction %", overlaying='y', side='right'),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.subheader("🔄 Tab Switches")
            if st.session_state.tab_switches:
                df_tabs = pd.DataFrame(st.session_state.tab_switches[-20:])
                if not df_tabs.empty:
                    fig = px.bar(
                        df_tabs,
                        x='timestamp',
                        y='tab_count',
                        title="Tab Switch Count Over Time",
                        labels={'tab_count': 'Tab Count', 'timestamp': 'Time'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Alerts section
        st.markdown("---")
        st.subheader("🚨 Alerts")
        if st.session_state.alerts:
            for alert in st.session_state.alerts[-10:]:
                timestamp = alert.get('timestamp', datetime.now())
                message = alert.get('message', 'Alert')
                severity = alert.get('severity', 'info')
                
                if severity == 'error':
                    st.error(f"[{timestamp.strftime('%H:%M:%S')}] {message}")
                elif severity == 'warning':
                    st.warning(f"[{timestamp.strftime('%H:%M:%S')}] {message}")
                else:
                    st.info(f"[{timestamp.strftime('%H:%M:%S')}] {message}")
        else:
            st.info("No alerts yet")
        
        # Auto-refresh - use a more efficient approach
        # Check if we should refresh based on data availability
        has_new_data = (
            not st.session_state.frame_queue.empty() or 
            not st.session_state.metrics_queue.empty()
        )
        
        if has_new_data:
            # Refresh quickly when there's new data
            time.sleep(0.15)
        else:
            # Refresh slower when waiting for data
            time.sleep(0.3)
        
        st.rerun()

if __name__ == "__main__":
    main()

