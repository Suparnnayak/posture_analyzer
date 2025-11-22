"""
Eye Tracking for User Concentration Monitoring

Tracks if user is looking at screen or away using head pose and eye position.
Uses MediaPipe Face Mesh for accurate eye and head detection.
"""
import cv2
import mediapipe as mp
import numpy as np

class EyeTracker:
    """Track user's eye position to detect if looking at screen."""
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh for eye tracking."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Key face landmarks
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.NOSE_TIP = 4
        self.CHIN = 152
        self.FOREHEAD = 10
        
        # For head pose estimation
        # 3D model points (approximate face model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),   # Left mouth corner
            (150.0, -150.0, -125.0)     # Right mouth corner
        ])
        
        # Camera matrix (approximate, will be refined)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
    def calculate_eye_center(self, landmarks, eye_indices):
        """Calculate center point of eye."""
        eye_points = []
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            eye_points.append([landmark.x, landmark.y])
        
        if not eye_points:
            return None
        
        eye_points = np.array(eye_points)
        center = np.mean(eye_points, axis=0)
        return center
    
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR) to detect if eyes are open."""
        try:
            # Get vertical distances
            vertical_1 = np.linalg.norm([
                landmarks.landmark[eye_indices[1]].x - landmarks.landmark[eye_indices[5]].x,
                landmarks.landmark[eye_indices[1]].y - landmarks.landmark[eye_indices[5]].y
            ])
            vertical_2 = np.linalg.norm([
                landmarks.landmark[eye_indices[2]].x - landmarks.landmark[eye_indices[4]].x,
                landmarks.landmark[eye_indices[2]].y - landmarks.landmark[eye_indices[4]].y
            ])
            
            # Get horizontal distance
            horizontal = np.linalg.norm([
                landmarks.landmark[eye_indices[0]].x - landmarks.landmark[eye_indices[3]].x,
                landmarks.landmark[eye_indices[0]].y - landmarks.landmark[eye_indices[3]].y
            ])
            
            if horizontal == 0:
                return 0.0
            
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        except:
            return 0.0
    
    def estimate_head_pose(self, landmarks, frame_shape):
        """Estimate head pose (rotation) to detect if head is turned."""
        try:
            h, w = frame_shape[:2]
            
            # Initialize camera matrix if not done
            if self.camera_matrix is None:
                focal_length = w
                center = (w / 2, h / 2)
                self.camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float32)
            
            # Get 2D image points from landmarks
            image_points = np.array([
                [landmarks.landmark[self.NOSE_TIP].x * w, landmarks.landmark[self.NOSE_TIP].y * h],
                [landmarks.landmark[self.CHIN].x * w, landmarks.landmark[self.CHIN].y * h],
                [landmarks.landmark[33].x * w, landmarks.landmark[33].y * h],  # Left eye corner
                [landmarks.landmark[263].x * w, landmarks.landmark[263].y * h],  # Right eye corner
                [landmarks.landmark[61].x * w, landmarks.landmark[61].y * h],  # Left mouth
                [landmarks.landmark[291].x * w, landmarks.landmark[291].y * h]  # Right mouth
            ], dtype=np.float32)
            
            # Solve PnP to get rotation and translation
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return None
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Extract Euler angles
            # Pitch (nodding up/down), Yaw (turning left/right), Roll (tilting)
            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            singular = sy < 1e-6
            
            if not singular:
                x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = 0
            
            # Convert to degrees
            pitch = np.degrees(x)  # Nodding
            yaw = np.degrees(y)    # Turning left/right
            roll = np.degrees(z)   # Tilting
            
            # Normalize angles to -90 to 90 range for pitch and yaw
            # Pitch: normalize to -90 to 90
            while pitch > 90:
                pitch -= 180
            while pitch < -90:
                pitch += 180
            
            # Yaw: normalize to -90 to 90
            while yaw > 90:
                yaw -= 180
            while yaw < -90:
                yaw += 180
            
            return {
                "pitch": float(pitch),
                "yaw": float(yaw),
                "roll": float(roll),
                "rotation_matrix": rotation_matrix
            }
        except Exception as e:
            return None
    
    def calculate_gaze_direction(self, landmarks, frame_shape):
        """
        Calculate gaze direction using head pose, eye position, and iris position.
        
        Returns:
            dict: Gaze direction info with dynamic confidence
        """
        try:
            # 1. Estimate head pose (most important for detecting looking away)
            head_pose = self.estimate_head_pose(landmarks, frame_shape)
            
            # 2. Get eye centers
            left_eye_center = self.calculate_eye_center(landmarks, self.LEFT_EYE_INDICES)
            right_eye_center = self.calculate_eye_center(landmarks, self.RIGHT_EYE_INDICES)
            
            if left_eye_center is None or right_eye_center is None:
                return {"looking_away": False, "confidence": 0.0, "reason": "no_eyes"}
            
            # 3. Calculate eye aspect ratio (EAR) to check if eyes are open
            left_ear = self.calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE_INDICES)
            right_ear = self.calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE_INDICES)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # 4. Get iris positions if available
            has_iris = len(landmarks.landmark) > 468
            if has_iris:
                left_iris = landmarks.landmark[468]
                right_iris = landmarks.landmark[473]
                
                # Calculate iris offset from eye center
                left_iris_offset_x = left_iris.x - left_eye_center[0]
                left_iris_offset_y = left_iris.y - left_eye_center[1]
                right_iris_offset_x = right_iris.x - right_eye_center[0]
                right_iris_offset_y = right_iris.y - right_eye_center[1]
                
                avg_iris_offset_x = (left_iris_offset_x + right_iris_offset_x) / 2
                avg_iris_offset_y = (left_iris_offset_y + right_iris_offset_y) / 2
                iris_distance = np.sqrt(avg_iris_offset_x**2 + avg_iris_offset_y**2)
            else:
                iris_distance = 0.0
                avg_iris_offset_x = 0.0
                avg_iris_offset_y = 0.0
            
            # 5. Calculate face center and eye position relative to face
            nose_tip = landmarks.landmark[self.NOSE_TIP]
            face_center_x = nose_tip.x
            face_center_y = nose_tip.y
            
            eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
            eye_center_y = (left_eye_center[1] + right_eye_center[1]) / 2
            
            eye_offset_x = abs(eye_center_x - face_center_x)
            eye_offset_y = abs(eye_center_y - face_center_y)
            eye_offset_distance = np.sqrt(eye_offset_x**2 + eye_offset_y**2)
            
            # 6. Determine if looking away based on multiple factors
            # Use a voting system - need strong evidence to mark as "looking away"
            looking_away_votes = 0
            confidence_factors = []
            
            # Factor 1: Head yaw (turning left/right) - most important
            if head_pose:
                yaw_abs = abs(head_pose["yaw"])
                pitch_abs = abs(head_pose["pitch"])
                
                # Only mark as looking away if head is turned significantly
                # Increased threshold to reduce false positives
                if yaw_abs > 25.0:  # Head turned more than 25 degrees (was 15)
                    looking_away_votes += 2  # Strong evidence
                    confidence_factors.append(min(1.0, yaw_abs / 50.0))
                elif yaw_abs > 20.0:  # Head turned more than 20 degrees (was 10)
                    looking_away_votes += 1  # Moderate evidence
                    confidence_factors.append(min(0.6, yaw_abs / 35.0))
                
                # Pitch (nodding) - only if very significant
                if pitch_abs > 30.0:  # Looking up or down significantly (was 20)
                    looking_away_votes += 1
                    confidence_factors.append(min(0.7, pitch_abs / 50.0))
            else:
                # Fallback: use eye position if head pose fails
                if eye_offset_distance > 0.10 or iris_distance > 0.08:  # Increased thresholds
                    looking_away_votes += 1
                    confidence_factors.append(min(0.5, max(eye_offset_distance, iris_distance) / 0.18))
            
            # Factor 2: Iris position (if available) - only if significant
            if has_iris and iris_distance > 0.08:  # Increased threshold (was 0.05)
                looking_away_votes += 1
                confidence_factors.append(min(0.6, iris_distance / 0.15))
            
            # Factor 3: Eye position relative to face center - only if significant
            if eye_offset_distance > 0.08:  # Increased threshold (was 0.06)
                looking_away_votes += 1
                confidence_factors.append(min(0.4, eye_offset_distance / 0.15))
            
            # Factor 4: Eyes closed (low EAR) - doesn't mean looking away, just lower confidence
            if avg_ear < 0.2:  # Eyes might be closed
                confidence_factors.append(0.2)  # Lower confidence when eyes closed
            
            # Need at least 2 votes to mark as "looking away" (reduces false positives)
            looking_away = looking_away_votes >= 2
            
            # Calculate dynamic confidence based on all factors
            if looking_away:
                # When looking away, confidence should be based on how strong the evidence is
                if confidence_factors:
                    # Average of all confidence factors
                    base_confidence = np.mean(confidence_factors)
                    
                    # Boost confidence if multiple factors agree and we have strong votes
                    if looking_away_votes >= 2 and len(confidence_factors) > 1:
                        base_confidence = min(1.0, base_confidence * 1.1)
                    
                    # Adjust based on EAR (eyes open = higher confidence in detection)
                    ear_factor = min(1.0, avg_ear / 0.3)  # Normalize EAR
                    confidence = base_confidence * (0.6 + 0.4 * ear_factor)
                else:
                    confidence = 0.3  # Low confidence if no factors
            else:
                # When looking at screen, confidence should be low (we're confident you're NOT looking away)
                # Calculate a "looking at screen" confidence based on how centered everything is
                if head_pose:
                    yaw_abs = abs(head_pose["yaw"])
                    pitch_abs = abs(head_pose["pitch"])
                    
                    # Lower confidence when head is very centered (high confidence you're looking at screen)
                    yaw_factor = max(0.0, 1.0 - (yaw_abs / 10.0))  # 1.0 when yaw=0, 0.0 when yaw>10
                    pitch_factor = max(0.0, 1.0 - (pitch_abs / 15.0))  # 1.0 when pitch=0, 0.0 when pitch>15
                    
                    # Confidence that you're looking at screen (inverse of looking away confidence)
                    confidence = 0.1 + 0.2 * (yaw_factor + pitch_factor) / 2.0
                else:
                    confidence = 0.15  # Default low confidence when looking at screen
            
            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                "looking_away": bool(looking_away),
                "confidence": float(confidence),
                "horizontal_offset": float(avg_iris_offset_x if has_iris else eye_offset_x),
                "vertical_offset": float(avg_iris_offset_y if has_iris else eye_offset_y),
                "distance": float(iris_distance if has_iris else eye_offset_distance),
                "head_yaw": float(head_pose["yaw"]) if head_pose else 0.0,
                "head_pitch": float(head_pose["pitch"]) if head_pose else 0.0,
                "eye_aspect_ratio": float(avg_ear)
            }
            
        except Exception as e:
            # Fallback to simple method
            try:
                left_eye_center = self.calculate_eye_center(landmarks, self.LEFT_EYE_INDICES)
                right_eye_center = self.calculate_eye_center(landmarks, self.RIGHT_EYE_INDICES)
                nose_tip = landmarks.landmark[self.NOSE_TIP]
                
                if left_eye_center is None or right_eye_center is None:
                    return {"looking_away": False, "confidence": 0.0, "error": "no_eyes"}
                
                eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
                eye_center_y = (left_eye_center[1] + right_eye_center[1]) / 2
                
                horizontal_offset = abs(eye_center_x - nose_tip.x)
                vertical_offset = abs(eye_center_y - nose_tip.y)
                
                is_looking_away = (horizontal_offset > 0.06) or (vertical_offset > 0.05)
                distance = np.sqrt(horizontal_offset**2 + vertical_offset**2)
                confidence = min(0.5, distance / 0.12)  # Lower confidence for fallback
                
                return {
                    "looking_away": bool(is_looking_away),
                    "confidence": float(confidence),
                    "horizontal_offset": float(horizontal_offset),
                    "vertical_offset": float(vertical_offset),
                    "distance": float(distance),
                    "head_yaw": 0.0,
                    "head_pitch": 0.0,
                    "eye_aspect_ratio": 0.0
                }
            except:
                return {"looking_away": False, "confidence": 0.0, "error": str(e)}
    
    def process_frame(self, frame):
        """
        Process frame and detect eye position.
        
        Args:
            frame: OpenCV frame (BGR)
            
        Returns:
            dict: Eye tracking results or None
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Detect looking away with dynamic confidence
        gaze_info = self.calculate_gaze_direction(face_landmarks, frame.shape)
        
        return {
            "looking_away": bool(gaze_info["looking_away"]),
            "confidence": float(gaze_info.get("confidence", 0.0)),
            "horizontal_offset": float(gaze_info.get("horizontal_offset", 0.0)),
            "vertical_offset": float(gaze_info.get("vertical_offset", 0.0)),
            "head_yaw": float(gaze_info.get("head_yaw", 0.0)),
            "head_pitch": float(gaze_info.get("head_pitch", 0.0)),
            "eye_aspect_ratio": float(gaze_info.get("eye_aspect_ratio", 0.0)),
            "landmarks": face_landmarks
        }
    
    def draw_face_mesh(self, frame, landmarks):
        """Draw face mesh on frame."""
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_face_mesh.FACEMESH_CONTOURS,
            None,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        )
