"""
Real-World Demo: Combined Posture + Tab Switch Monitoring

This script runs both posture detection and tab switch monitoring simultaneously.
"""
import threading
import time
from realtime_posture_detector import PostureDetector
from tab_switch_monitor import TabSwitchMonitor

class CombinedMonitor:
    """Combined posture and tab switch monitoring."""
    
    def __init__(self, user_id=1):
        """Initialize combined monitor."""
        self.user_id = user_id
        self.posture_detector = PostureDetector()
        self.tab_monitor = TabSwitchMonitor()
        self.running = False
    
    def run_posture_detection(self):
        """Run posture detection in separate thread."""
        self.posture_detector.run(
            user_id=self.user_id,
            show_video=True,
            save_events=False
        )
        self.running = False
    
    def run_tab_monitoring(self):
        """Run tab monitoring in separate thread."""
        self.tab_monitor.monitor(
            user_id=self.user_id,
            check_interval=0.5
        )
        self.running = False
    
    def start(self):
        """Start both monitors."""
        self.running = True
        
        print("=" * 60)
        print("Real-World Focus Tracking Demo")
        print("=" * 60)
        print("\nStarting:")
        print("  1. Posture detection (webcam)")
        print("  2. Tab switch monitoring")
        print("\nPress Ctrl+C to stop")
        print("=" * 60)
        
        # Start posture detection in separate thread
        posture_thread = threading.Thread(target=self.run_posture_detection, daemon=True)
        posture_thread.start()
        
        # Start tab monitoring in separate thread
        tab_thread = threading.Thread(target=self.run_tab_monitoring, daemon=True)
        tab_thread.start()
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping all monitors...")
            self.running = False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-world focus tracking demo')
    parser.add_argument('--user-id', type=int, default=1, help='User ID')
    parser.add_argument('--posture-only', action='store_true', help='Only run posture detection')
    parser.add_argument('--tabs-only', action='store_true', help='Only run tab monitoring')
    
    args = parser.parse_args()
    
    if args.posture_only:
        detector = PostureDetector()
        detector.run(user_id=args.user_id)
    elif args.tabs_only:
        monitor = TabSwitchMonitor()
        monitor.monitor(user_id=args.user_id)
    else:
        # Run both
        combined = CombinedMonitor(user_id=args.user_id)
        combined.start()

if __name__ == "__main__":
    main()

