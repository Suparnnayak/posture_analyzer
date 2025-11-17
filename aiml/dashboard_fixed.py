"""
Fixed Dashboard - Better Camera and Tab Handling

This version fixes camera access issues by:
- Using proper camera backend (DirectShow on Windows)
- Keeping camera open continuously
- Better error recovery
- Proper thread management
"""
import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
import threading
import queue
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import platform
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from realtime_posture_detector import PostureDetector
from tab_switch_monitor import TabSwitchMonitor
from inference.predictor import FocusPredictor

# Page config
st.set_page_config(
    page_title="Focus Tracking Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'posture_data' not in st.session_state:
    st.session_state.posture_data = []
if 'tab_switches' not in st.session_state:
    st.session_state.tab_switches = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=5)  # Larger queue
if 'metrics_queue' not in st.session_state:
    st.session_state.metrics_queue = queue.Queue()
if 'camera_cap' not in st.session_state:
    st.session_state.camera_cap = None
if 'camera_thread' not in st.session_state:
    st.session_state.camera_thread = None

class EnhancedPostureDetector(PostureDetector):
    """Enhanced posture detector with looking away duration tracking."""
    
    def __init__(self):
        super().__init__()
        self.looking_away_start_time = None
        self.lookaway_threshold_seconds = 10.0
    
    def check_looking_away_duration(self, is_looking_away):
        """Check if user has been looking away for more than threshold."""
        current_time = time.time()
        
        if is_looking_away:
            if self.looking_away_start_time is None:
                self.looking_away_start_time = current_time
            
            duration = current_time - self.looking_away_start_time
            too_long = duration >= self.lookaway_threshold_seconds
            
            return True, too_long, duration
        else:
            if self.looking_away_start_time is not None:
                self.looking_away_start_time = None
            return False, False, 0.0

class EnhancedTabMonitor(TabSwitchMonitor):
    """Enhanced tab monitor with immediate alerts."""
    
    def set_alert_callback(self, callback):
        """Set callback function for tab switch alerts."""
        self.alert_callback = callback
    
    def create_tab_event(self, user_id=1):
        """Create tab switch event and trigger alert."""
        self.tab_count += 1
        current_time = time.time()
        self.switch_times.append(current_time)
        
        tab_frequency = self.calculate_tab_frequency()
        
        event = {
            "timestamp": int(current_time * 1000),
            "user_id": user_id,
            "eventType": "tab_switch",
            "duration": 0,
            "tabCount": self.tab_count,
            "tabFrequency": tab_frequency,
            "spineAngle": 0,
            "slouch": 0,
            "lookingAway": 0
        }
        
        # Trigger immediate alert
        if self.alert_callback:
            self.alert_callback({
                "type": "tab_switch",
                "message": f"Tab switched! (Total: {event['tabCount']})",
                "timestamp": datetime.now(),
                "severity": "warning"
            })
        
        return event

def camera_loop(detector, user_id, frame_queue, metrics_queue, camera_index=0):
    """Dedicated camera loop that keeps camera open."""
    # Use DirectShow backend on Windows for better compatibility
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    
    cap = cv2.VideoCapture(camera_index, backend)
    
    if not cap.isOpened():
        # Try without backend specification
        cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        metrics_queue.put({
            "error": "Could not open camera. Close other apps using camera and try again."
        })
        return
    
    # Set properties
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except:
        pass
    
    # Warm up camera - read a few frames
    for _ in range(5):
        cap.read()
        time.sleep(0.1)
    
    metrics_queue.put({
        "type": "info",
        "message": "Camera initialized successfully"
    })
    
    last_prediction_time = 0
    prediction_interval = 0.5
    consecutive_failures = 0
    
    try:
        while st.session_state.is_running:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    metrics_queue.put({
                        "type": "warning",
                        "message": "Camera read failures. Check if camera is still connected."
                    })
                    time.sleep(0.5)
                continue
            
            consecutive_failures = 0
            
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
                
                # Predict distraction (throttled)
                current_time = time.time()
                if detector.predictor and (current_time - last_prediction_time) >= prediction_interval:
                    prediction = detector.predictor.predict(event)
                    
                    event['distraction_prob'] = prediction['distraction_prob']
                    event['focus_prob'] = prediction['focus_prob']
                    event['is_distracted'] = prediction['is_distracted']
                    
                    # Check if should alert
                    if too_long:
                        metrics_queue.put({
                            "type": "alert",
                            "message": f"Looking away for {duration:.1f} seconds!",
                            "timestamp": datetime.now(),
                            "severity": "error"
                        })
                    
                    last_prediction_time = current_time
                
                # Convert frame for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Put frame in queue (non-blocking, drop old frames)
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()  # Remove oldest
                    except:
                        pass
                frame_queue.put(frame_rgb)
                
                # Put metrics
                metrics = {
                    "timestamp": datetime.now(),
                    "spine_angle": spine_angle,
                    "slouch": slouch,
                    "looking_away": looking_away,
                    "looking_away_duration": duration if is_away_now else 0.0,
                    "too_long": too_long,
                    "distraction_prob": event.get('distraction_prob', 0.0),
                    "focus_prob": event.get('focus_prob', 1.0)
                }
                metrics_queue.put(metrics)
            
            # Small delay to prevent overwhelming
            time.sleep(0.03)  # ~30 FPS
    
    except Exception as e:
        metrics_queue.put({"error": f"Camera error: {str(e)}"})
    finally:
        if cap is not None:
            cap.release()
        metrics_queue.put({
            "type": "info",
            "message": "Camera released"
        })

def tab_monitoring_loop(monitor, user_id, metrics_queue):
    """Tab monitoring loop."""
    def alert_callback(alert_data):
        metrics_queue.put(alert_data)
    
    monitor.set_alert_callback(alert_callback)
    
    # Check dependencies
    system = platform.system()
    can_monitor = False
    
    if system == "Windows":
        try:
            import win32gui
            can_monitor = True
        except ImportError:
            metrics_queue.put({
                "type": "warning",
                "message": "Tab monitoring: Install pywin32 (pip install pywin32)"
            })
    elif system == "Linux":
        import subprocess
        try:
            subprocess.run(['xdotool', '--version'], capture_output=True, check=True)
            can_monitor = True
        except:
            metrics_queue.put({
                "type": "warning",
                "message": "Tab monitoring: Install xdotool (sudo apt-get install xdotool)"
            })
    else:
        can_monitor = True  # Mac
    
    previous_title = None
    check_interval = 0.5
    
    try:
        while st.session_state.is_running:
            try:
                if can_monitor:
                    current_title = monitor.get_active_window_title()
                    
                    if monitor.detect_tab_switch(current_title, previous_title):
                        event = monitor.create_tab_event(user_id)
                        
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
                # Silently continue
                pass
            
            time.sleep(check_interval)
    
    except Exception as e:
        metrics_queue.put({"error": f"Tab monitoring error: {str(e)}"})

def main():
    """Main dashboard function."""
    st.title("🎯 Real-Time Focus Tracking Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        user_id = st.number_input("User ID", min_value=1, value=1, step=1)
        
        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("▶️ Start", type="primary", use_container_width=True)
        with col2:
            stop_btn = st.button("⏹️ Stop", use_container_width=True)
        
        if start_btn:
            if st.session_state.camera_cap is not None:
                st.warning("Already running! Click Stop first.")
            else:
                st.session_state.is_running = True
                st.rerun()
        
        if stop_btn:
            st.session_state.is_running = False
            if st.session_state.camera_cap is not None:
                st.session_state.camera_cap.release()
                st.session_state.camera_cap = None
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
        
        camera_index = st.number_input("Camera Index", min_value=0, max_value=3, value=0, step=1)
        
        st.markdown("---")
        st.markdown("### ℹ️ Info")
        st.info("""
        **Alerts:**
        - ⚠️ Tab switch: Immediate alert
        - ⚠️ Looking away >10s: Focus alert
        """)
    
    # Main content
    if not st.session_state.is_running:
        st.info("👆 Click 'Start' in the sidebar to begin monitoring")
        
        # Diagnostics
        with st.expander("🔍 System Diagnostics", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Camera Status")
                test_cap = cv2.VideoCapture(camera_index)
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
                test_cap.release()
            
            with col2:
                st.subheader("Tab Monitoring Status")
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
        
        if st.session_state.posture_data:
            st.subheader("📈 Recent Data")
            df = pd.DataFrame(st.session_state.posture_data[-50:])
            st.line_chart(df[['spine_angle', 'distraction_prob']])
    
    else:
        # Initialize detectors
        if 'detector' not in st.session_state:
            try:
                st.session_state.detector = EnhancedPostureDetector()
                st.session_state.detector.lookaway_threshold_seconds = lookaway_threshold
            except Exception as e:
                st.error(f"Failed to initialize detector: {e}")
                st.stop()
        
        if 'tab_monitor' not in st.session_state:
            try:
                st.session_state.tab_monitor = EnhancedTabMonitor()
            except Exception as e:
                st.warning(f"Tab monitor warning: {e}")
        
        # Start threads only once
        if 'threads_started' not in st.session_state:
            st.session_state.threads_started = True
            
            # Start camera thread
            camera_thread = threading.Thread(
                target=camera_loop,
                args=(
                    st.session_state.detector,
                    user_id,
                    st.session_state.frame_queue,
                    st.session_state.metrics_queue,
                    camera_index
                ),
                daemon=True
            )
            camera_thread.start()
            st.session_state.camera_thread = camera_thread
            
            # Start tab monitoring thread
            tab_thread = threading.Thread(
                target=tab_monitoring_loop,
                args=(
                    st.session_state.tab_monitor,
                    user_id,
                    st.session_state.metrics_queue
                ),
                daemon=True
            )
            tab_thread.start()
        
        # Main layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Live Video Feed")
            video_placeholder = st.empty()
            
            # Get latest frame
            try:
                frame = st.session_state.frame_queue.get_nowait()
                video_placeholder.image(frame, channels="RGB", use_container_width=True)
            except queue.Empty:
                # Check for errors
                error_found = False
                temp_items = []
                while not st.session_state.metrics_queue.empty():
                    try:
                        item = st.session_state.metrics_queue.get_nowait()
                        if "error" in item:
                            error_found = True
                            video_placeholder.error(f"❌ {item['error']}")
                        else:
                            temp_items.append(item)
                    except:
                        break
                
                # Put items back
                for item in temp_items:
                    st.session_state.metrics_queue.put(item)
                
                if not error_found:
                    video_placeholder.info("⏳ Initializing camera... (this may take 5-10 seconds)")
        
        with col2:
            st.subheader("📊 Current Metrics")
            
            # Process metrics
            metrics_data = {}
            while not st.session_state.metrics_queue.empty():
                try:
                    metric = st.session_state.metrics_queue.get_nowait()
                    
                    if metric.get("type") == "alert":
                        st.session_state.alerts.append(metric)
                        st.error(f"⚠️ {metric['message']}")
                    elif metric.get("type") == "tab_switch":
                        st.session_state.tab_switches.append(metric)
                        st.warning(f"🔄 Tab Switch #{metric.get('tab_count', 0)}")
                    elif metric.get("type") == "warning":
                        st.warning(f"⚠️ {metric.get('message', 'Warning')}")
                    elif metric.get("type") == "info":
                        st.info(f"ℹ️ {metric.get('message', 'Info')}")
                    elif "error" in metric:
                        st.error(f"❌ {metric['error']}")
                    else:
                        metrics_data = metric
                        st.session_state.posture_data.append(metric)
                except:
                    break
            
            # Display metrics
            if metrics_data:
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Spine Angle", f"{metrics_data.get('spine_angle', 0):.1f}°")
                with m2:
                    st.metric("Distraction", f"{metrics_data.get('distraction_prob', 0)*100:.1f}%")
                with m3:
                    st.metric("Focus", f"{metrics_data.get('focus_prob', 1)*100:.1f}%")
                
                if metrics_data.get('looking_away_duration', 0) > 0:
                    duration = metrics_data['looking_away_duration']
                    if metrics_data.get('too_long', False):
                        st.error(f"⚠️ Looking Away: {duration:.1f}s (NOT FOCUSED!)")
                    else:
                        st.warning(f"👀 Looking Away: {duration:.1f}s")
            else:
                st.info("Waiting for data...")
        
        # Charts
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
                        title="Tab Switch Count",
                        labels={'tab_count': 'Tab Count', 'timestamp': 'Time'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Alerts
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
        
        # Auto-refresh (slower to reduce CPU)
        time.sleep(0.15)
        st.rerun()

if __name__ == "__main__":
    main()

