"""
Tab Switch Monitor

Monitors browser tab switching events and sends them to AIML predictor.
Works on Windows, Linux, and Mac.
"""
import time
import platform
from datetime import datetime
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from inference.predictor import FocusPredictor
except ImportError:
    FocusPredictor = None

class TabSwitchMonitor:
    """Monitor tab switching events."""
    
    def __init__(self):
        """Initialize tab switch monitor."""
        self.system = platform.system()
        self.tab_count = 0
        self.last_switch_time = time.time()
        self.switch_times = []
        self.alert_callback = None  # Callback for alerts
        
        # Initialize AIML predictor
        try:
            if FocusPredictor:
                self.predictor = FocusPredictor()
                print("[OK] AIML predictor loaded")
            else:
                self.predictor = None
        except Exception as e:
            print(f"[WARNING] Could not load AIML predictor: {e}")
            self.predictor = None
    
    def set_alert_callback(self, callback):
        """Set callback function for tab switch alerts."""
        self.alert_callback = callback
    
    def get_active_window_title(self):
        """Get active window title (browser tab)."""
        try:
            if self.system == "Windows":
                import win32gui
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd)
                return title
            elif self.system == "Linux":
                import subprocess
                result = subprocess.run(
                    ['xdotool', 'getactivewindow', 'getwindowname'],
                    capture_output=True,
                    text=True
                )
                return result.stdout.strip()
            elif self.system == "Darwin":  # Mac
                import subprocess
                result = subprocess.run(
                    ['osascript', '-e', 'tell application "System Events" to get name of first process whose frontmost is true'],
                    capture_output=True,
                    text=True
                )
                return result.stdout.strip()
        except Exception as e:
            print(f"[WARNING] Could not get window title: {e}")
        return None
    
    def detect_tab_switch(self, current_title, previous_title):
        """Detect if tab was switched."""
        if not current_title or not previous_title:
            return False
        
        # Check if it's a browser window
        browsers = ['chrome', 'firefox', 'edge', 'safari', 'opera', 'brave']
        is_browser = any(browser in current_title.lower() for browser in browsers)
        
        if not is_browser:
            return False
        
        # Check if title changed (tab switch)
        return current_title != previous_title
    
    def calculate_tab_frequency(self):
        """Calculate tab switch frequency."""
        if not self.switch_times:
            return 0.0
        
        # Keep only last 60 seconds
        current_time = time.time()
        recent_switches = [t for t in self.switch_times if current_time - t < 60]
        self.switch_times = recent_switches
        
        # Frequency = switches per minute
        if len(recent_switches) < 2:
            return 0.0
        
        time_span = recent_switches[-1] - recent_switches[0]
        if time_span == 0:
            return 0.0
        
        frequency = (len(recent_switches) - 1) / (time_span / 60.0)
        return min(1.0, frequency / 10.0)  # Normalize to 0-1
    
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
    
    def monitor(self, user_id=1, check_interval=0.5):
        """
        Monitor tab switching in real-time.
        
        Args:
            user_id: User ID
            check_interval: How often to check for tab switches (seconds)
        """
        print(f"[OK] Starting tab switch monitor (checking every {check_interval}s)")
        print("Press Ctrl+C to stop")
        
        previous_title = None
        
        try:
            while True:
                current_title = self.get_active_window_title()
                
                if self.detect_tab_switch(current_title, previous_title):
                    event = self.create_tab_event(user_id)
                    
                    # Predict distraction
                    if self.predictor:
                        prediction = self.predictor.predict(event)
                        
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Tab Switch #{event['tabCount']} | "
                              f"Frequency: {event['tabFrequency']:.2f} | "
                              f"Distraction: {prediction['distraction_prob']:.1%} | "
                              f"Status: {'DISTRACTED' if prediction['is_distracted'] else 'FOCUSED'}")
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Tab Switch #{event['tabCount']} | "
                              f"Frequency: {event['tabFrequency']:.2f}")
                
                previous_title = current_title
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print("\n[INFO] Stopping monitor...")
        except Exception as e:
            print(f"[ERROR] {e}")
            print("\n[INFO] Make sure required dependencies are installed:")
            if self.system == "Windows":
                print("  pip install pywin32")
            elif self.system == "Linux":
                print("  sudo apt-get install xdotool")
            # Mac doesn't need extra dependencies

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor tab switching')
    parser.add_argument('--user-id', type=int, default=1, help='User ID')
    parser.add_argument('--interval', type=float, default=0.5, help='Check interval in seconds')
    
    args = parser.parse_args()
    
    monitor = TabSwitchMonitor()
    monitor.monitor(user_id=args.user_id, check_interval=args.interval)

if __name__ == "__main__":
    main()

