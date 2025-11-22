"""
Posture Analyzer for User Interest Detection

Analyzes user posture to determine interest/engagement level.
Uses MediaPipe Pose for upper body posture (chest up only).
"""
import cv2
import mediapipe as mp
import numpy as np

class PostureAnalyzer:
    """Analyze user posture to determine interest level."""
    
    def __init__(self):
        """Initialize MediaPipe Pose."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def calculate_neck_angle(self, landmarks):
        """
        Calculate neck/upper spine angle from pose landmarks.
        For upper body only (chest up visible).
        Returns angle from vertical (0° = upright, 90° = horizontal).
        """
        try:
            # Use shoulders and head position (since only chest up is visible)
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            
            # Calculate shoulder midpoint
            shoulder_mid = np.array([
                (left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2
            ])
            
            # Head position (using nose as proxy)
            head_pos = np.array([nose.x, nose.y])
            
            # Calculate vector from shoulders to head
            neck_vector = head_pos - shoulder_mid
            
            # Calculate angle from vertical
            vertical_component = abs(neck_vector[1])  # Y component (height)
            horizontal_component = abs(neck_vector[0])  # X component (width)
            
            if vertical_component == 0:
                return 90.0  # Completely horizontal
            
            # Calculate angle: atan(horizontal/vertical)
            angle_rad = np.arctan2(horizontal_component, vertical_component)
            angle_deg = np.degrees(angle_rad)
            
            # Clamp to 0-90 degrees
            return min(90.0, max(0.0, angle_deg))
        except Exception as e:
            return 0.0
    
    def calculate_spine_angle(self, landmarks):
        """
        Calculate spine angle from pose landmarks (fallback if hips visible).
        Returns angle from vertical (0° = upright, 90° = horizontal).
        """
        try:
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Check if hips are visible (confidence > 0)
            if left_hip.visibility < 0.5 or right_hip.visibility < 0.5:
                # Hips not visible, use neck angle instead
                return self.calculate_neck_angle(landmarks)
            
            # Calculate midpoints
            shoulder_mid = np.array([
                (left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2
            ])
            
            hip_mid = np.array([
                (left_hip.x + right_hip.x) / 2,
                (left_hip.y + right_hip.y) / 2
            ])
            
            # Calculate vector from hip to shoulder
            spine_vector = shoulder_mid - hip_mid
            
            # Calculate angle from vertical
            vertical_component = abs(spine_vector[1])  # Y component (height)
            horizontal_component = abs(spine_vector[0])  # X component (width)
            
            if vertical_component == 0:
                return 90.0  # Completely horizontal
            
            # Calculate angle: atan(horizontal/vertical)
            angle_rad = np.arctan2(horizontal_component, vertical_component)
            angle_deg = np.degrees(angle_rad)
            
            # Clamp to 0-90 degrees
            return min(90.0, max(0.0, angle_deg))
        except Exception as e:
            return 0.0
    
    def detect_slouch(self, angle, threshold=25.0):
        """
        Detect if user is slouching.
        Adjusted threshold for upper body only.
        """
        return angle > threshold
    
    def calculate_interest_level(self, spine_angle, slouch, visibility_score=1.0):
        """
        Calculate user interest level based on posture.
        
        Args:
            spine_angle: Angle from vertical
            slouch: Whether user is slouching
            visibility_score: How well landmarks are visible (0-1)
        
        Returns:
            dict: Interest level and score
        """
        # Calculate base score from spine angle
        # Adjusted for upper body only - most people have 5-15° when engaged
        # 0-5° = excellent (0.8), 5-12° = good (0.65), 12-20° = fair (0.45), 20-30° = poor (0.25), 30+° = very poor (0.1)
        if spine_angle <= 5.0:
            angle_score = 0.8  # Very good posture
        elif spine_angle <= 12.0:
            # Linear interpolation: 0.8 at 5°, 0.65 at 12°
            angle_score = 0.8 - ((spine_angle - 5.0) / 7.0) * 0.15
        elif spine_angle <= 20.0:
            # Linear interpolation: 0.65 at 12°, 0.45 at 20°
            angle_score = 0.65 - ((spine_angle - 12.0) / 8.0) * 0.2
        elif spine_angle <= 30.0:
            # Linear interpolation: 0.45 at 20°, 0.25 at 30°
            angle_score = 0.45 - ((spine_angle - 20.0) / 10.0) * 0.2
        else:
            # Linear interpolation: 0.25 at 30°, 0.1 at 90°
            angle_score = max(0.05, 0.25 - ((spine_angle - 30.0) / 60.0) * 0.15)
        
        # Slouch penalty - apply significant penalty for slouching
        slouch_penalty = 0.25 if slouch and spine_angle > 20.0 else 0.12 if slouch else 0.0
        
        # Calculate interest score (0-1)
        interest_score = max(0.05, min(0.8, angle_score - slouch_penalty))
        
        # Adjust for visibility (lower confidence if landmarks not well visible)
        interest_score = interest_score * (0.7 + 0.3 * visibility_score)
        
        # Categorize interest level
        if interest_score >= 0.55:
            interest_level = "high"
        elif interest_score >= 0.3:
            interest_level = "medium"
        else:
            interest_level = "low"
        
        return {
            "interest_score": float(interest_score),
            "interest_level": str(interest_level),
            "spine_angle": float(spine_angle),
            "slouch": bool(slouch)
        }
    
    def calculate_visibility_score(self, landmarks):
        """Calculate how well key landmarks are visible (0-1)."""
        try:
            key_landmarks = [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.NOSE
            ]
            
            visibilities = []
            for landmark_idx in key_landmarks:
                visibilities.append(landmarks.landmark[landmark_idx].visibility)
            
            return float(np.mean(visibilities))
        except:
            return 0.5  # Default moderate visibility
    
    def process_frame(self, frame):
        """
        Process frame and analyze posture.
        
        Args:
            frame: OpenCV frame (BGR)
            
        Returns:
            dict: Posture analysis results or None
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Calculate visibility score
        visibility_score = self.calculate_visibility_score(results.pose_landmarks)
        
        # Calculate spine angle (will use neck angle if hips not visible)
        spine_angle = self.calculate_spine_angle(results.pose_landmarks)
        
        # Detect slouch
        slouch = self.detect_slouch(spine_angle)
        
        # Calculate interest level
        interest = self.calculate_interest_level(spine_angle, slouch, visibility_score)
        
        return {
            "spine_angle": spine_angle,
            "slouch": slouch,
            "interest_score": interest["interest_score"],
            "interest_level": interest["interest_level"],
            "visibility_score": visibility_score,
            "landmarks": results.pose_landmarks
        }
