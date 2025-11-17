"""
Real-Time Posture Detection using Webcam

Uses MediaPipe Pose for real-time posture detection from webcam feed.
Calculates spine angle and detects slouching.
"""
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
from inference.predictor import FocusPredictor
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

class PostureDetector:
    """Real-time posture detection using MediaPipe."""
    
    def __init__(self):
        """Initialize MediaPipe pose detection."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize AIML predictor
        try:
            self.predictor = FocusPredictor()
            print("[OK] AIML predictor loaded")
        except Exception as e:
            print(f"[WARNING] Could not load AIML predictor: {e}")
            self.predictor = None
    
    def calculate_spine_angle(self, landmarks):
        """
        Calculate spine angle from pose landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            float: Spine angle in degrees (0-90)
        """
        # Get key points
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate midpoints
        shoulder_mid = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        ])
        
        hip_mid = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        ])
        
        # Calculate angle
        # Vertical = 0 degrees, leaning forward = higher angle
        delta_y = abs(shoulder_mid[1] - hip_mid[1])
        delta_x = abs(shoulder_mid[0] - hip_mid[0])
        
        if delta_x == 0:
            return 90.0
        
        angle_rad = np.arctan2(delta_y, delta_x)
        angle_deg = np.degrees(angle_rad)
        
        return min(90.0, max(0.0, angle_deg))
    
    def detect_slouch(self, angle, threshold=25.0):
        """Detect if user is slouching based on spine angle."""
        return 1 if angle > threshold else 0
    
    def detect_looking_away(self, landmarks):
        """
        Detect if user is looking away based on head position.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            int: 1 if looking away, 0 otherwise
        """
        try:
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Calculate shoulder midpoint
            shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
            
            # Calculate horizontal offset
            horizontal_offset = abs(nose.x - shoulder_mid_x)
            
            # Threshold for looking away
            threshold = 0.15  # Adjust based on testing
            
            return 1 if horizontal_offset > threshold else 0
        except:
            return 0
    
    def process_frame(self, frame, user_id=1):
        """
        Process a single frame and return posture data.
        
        Args:
            frame: OpenCV frame (BGR)
            user_id: User ID for prediction
            
        Returns:
            dict: Posture event data or None if no pose detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Calculate spine angle
        spine_angle = self.calculate_spine_angle(results.pose_landmarks)
        
        # Detect slouch
        slouch = self.detect_slouch(spine_angle)
        
        # Detect looking away
        looking_away = self.detect_looking_away(results.pose_landmarks)
        
        # Create event
        event = {
            "timestamp": int(time.time() * 1000),
            "user_id": user_id,
            "spineAngle": float(spine_angle),
            "slouch": slouch,
            "lookingAway": looking_away,
            "eventType": "posture",
            "duration": 0
        }
        
        return event, results.pose_landmarks
    
    def draw_pose(self, frame, landmarks, spine_angle, slouch, looking_away):
        """Draw pose landmarks and info on frame."""
        # Draw pose landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        
        # Draw text info
        cv2.putText(frame, f"Spine Angle: {spine_angle:.1f}°", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        slouch_text = "Slouching: YES" if slouch else "Slouching: NO"
        slouch_color = (0, 0, 255) if slouch else (0, 255, 0)
        cv2.putText(frame, slouch_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, slouch_color, 2)
        
        lookaway_text = "Looking Away: YES" if looking_away else "Looking Away: NO"
        lookaway_color = (0, 0, 255) if looking_away else (0, 255, 0)
        cv2.putText(frame, lookaway_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, lookaway_color, 2)
        
        return frame
    
    def run(self, user_id=1, show_video=True, save_events=False):
        """
        Run real-time posture detection.
        
        Args:
            user_id: User ID
            show_video: Show video feed
            save_events: Save events to file
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Could not open webcam")
            return
        
        print("[OK] Webcam opened")
        print("Press 'q' to quit")
        
        last_prediction_time = 0
        prediction_interval = 0.5  # Predict every 0.5 seconds
        
        events_log = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = self.process_frame(frame, user_id)
                
                if result:
                    event, landmarks = result
                    spine_angle = event["spineAngle"]
                    slouch = event["slouch"]
                    looking_away = event["lookingAway"]
                    
                    # Draw pose
                    if show_video:
                        frame = self.draw_pose(frame, landmarks, spine_angle, slouch, looking_away)
                    
                    # Predict distraction (throttled)
                    current_time = time.time()
                    if self.predictor and (current_time - last_prediction_time) >= prediction_interval:
                        prediction = self.predictor.predict(event)
                        
                        # Display prediction
                        if show_video:
                            dist_prob = prediction['distraction_prob']
                            focus_prob = prediction['focus_prob']
                            is_distracted = prediction['is_distracted']
                            
                            cv2.putText(frame, f"Distraction: {dist_prob:.1%}", (10, 120),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            cv2.putText(frame, f"Focus: {focus_prob:.1%}", (10, 150),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            
                            status = "DISTRACTED" if is_distracted else "FOCUSED"
                            status_color = (0, 0, 255) if is_distracted else (0, 255, 0)
                            cv2.putText(frame, f"Status: {status}", (10, 180),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                        
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Angle: {spine_angle:.1f}° | "
                              f"Slouch: {slouch} | "
                              f"LookAway: {looking_away} | "
                              f"Distraction: {prediction['distraction_prob']:.1%}")
                        
                        last_prediction_time = current_time
                    
                    # Save event
                    if save_events:
                        events_log.append(event)
                
                # Show frame
                if show_video:
                    cv2.imshow('Posture Detection', frame)
                
                # Quit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n[INFO] Stopping...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Save events if requested
            if save_events and events_log:
                import pandas as pd
                df = pd.DataFrame(events_log)
                filename = f"posture_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                print(f"[OK] Saved {len(events_log)} events to {filename}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time posture detection')
    parser.add_argument('--user-id', type=int, default=1, help='User ID')
    parser.add_argument('--no-video', action='store_true', help='Hide video feed')
    parser.add_argument('--save', action='store_true', help='Save events to CSV')
    
    args = parser.parse_args()
    
    detector = PostureDetector()
    detector.run(
        user_id=args.user_id,
        show_video=not args.no_video,
        save_events=args.save
    )

if __name__ == "__main__":
    main()

