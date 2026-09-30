import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load trained model and label encoder
model = joblib.load("models/exercise_classifier.pkl")
le = joblib.load("models/label_encoder.pkl")

# Configuration
WINDOW_SIZE = 10  # Number of frames to average predictions over
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to display prediction
SMOOTHING_FACTOR = 0.3  # Exponential smoothing factor (0-1)

class ExercisePredictor:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.prediction_history = deque(maxlen=WINDOW_SIZE)
        self.current_prediction = None
        self.last_prediction_time = time.time()
        self.smoothed_probs = None

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    def extract_features(self, landmarks):
        """Extract pose landmarks with additional features"""
        features = []
        
        # Basic landmarks
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        # Calculate additional features if we have enough landmarks
        if len(landmarks) >= 32:
            # Key distances
            left_shoulder = [landmarks[11].x, landmarks[11].y]
            right_shoulder = [landmarks[12].x, landmarks[12].y]
            shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
            features.append(shoulder_width)
            
            # Torso length (shoulder to hip)
            torso_length = np.linalg.norm(np.array([landmarks[11].x, landmarks[11].y]) - 
                                         np.array([landmarks[23].x, landmarks[23].y]))
            features.append(torso_length)
            
            # Key angles
            features.append(self.calculate_angle(
                [landmarks[11].x, landmarks[11].y],
                [landmarks[13].x, landmarks[13].y],
                [landmarks[15].x, landmarks[15].y]
            ))  # Left elbow
            
            features.append(self.calculate_angle(
                [landmarks[12].x, landmarks[12].y],
                [landmarks[14].x, landmarks[14].y],
                [landmarks[16].x, landmarks[16].y]
            ))  # Right elbow
            
            features.append(self.calculate_angle(
                [landmarks[23].x, landmarks[23].y],
                [landmarks[25].x, landmarks[25].y],
                [landmarks[27].x, landmarks[27].y]
            ))  # Left knee
            
            features.append(self.calculate_angle(
                [landmarks[24].x, landmarks[24].y],
                [landmarks[26].x, landmarks[26].y],
                [landmarks[28].x, landmarks[28].y]
            ))  # Right knee
        
        return features

    def predict_exercise(self, frame):
        """Process frame and predict exercise"""
        # Convert to RGB and process with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        if results.pose_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extract features
            features = self.extract_features(results.pose_landmarks.landmark)
            
            if len(features) == 132 + 6:  # 132 landmarks + 6 additional features
                # Get prediction probabilities
                current_probs = model.predict_proba([features])[0]
                
                # Initialize smoothed probabilities if needed
                if self.smoothed_probs is None:
                    self.smoothed_probs = current_probs
                else:
                    # Apply exponential smoothing
                    self.smoothed_probs = (SMOOTHING_FACTOR * current_probs + 
                                         (1 - SMOOTHING_FACTOR) * self.smoothed_probs)
                
                # Get best prediction
                pred_index = np.argmax(self.smoothed_probs)
                confidence = self.smoothed_probs[pred_index]
                exercise = le.inverse_transform([pred_index])[0]
                
                # Only update prediction if confidence is high enough
                if confidence > CONFIDENCE_THRESHOLD:
                    self.current_prediction = (exercise, confidence)
                    self.last_prediction_time = time.time()
                elif time.time() - self.last_prediction_time > 2:  # Reset after 2 seconds of low confidence
                    self.current_prediction = None
                
                return self.current_prediction, results.pose_landmarks
        
        return None, results.pose_landmarks if results.pose_landmarks else None

def main():
    cap = cv2.VideoCapture(0)
    predictor = ExercisePredictor()
    
    # Exercise-specific variables
    rep_count = 0
    direction = 0  # 0 = down, 1 = up
    last_exercise = None
    performance_metrics = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        
        # Predict exercise
        prediction, landmarks = predictor.predict_exercise(frame)
        
        if prediction:
            exercise, confidence = prediction
            
            # Display prediction
            cv2.putText(frame, f"{exercise} ({confidence:.1%})", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Exercise-specific feedback
            if landmarks and len(landmarks.landmark) >= 32:
                if exercise == "squat":
                    # Get right leg landmarks
                    hip = [landmarks.landmark[24].x, landmarks.landmark[24].y]
                    knee = [landmarks.landmark[26].x, landmarks.landmark[26].y]
                    ankle = [landmarks.landmark[28].x, landmarks.landmark[28].y]
                    
                    angle = predictor.calculate_angle(hip, knee, ankle)
                    cv2.putText(frame, f"Knee Angle: {int(angle)}°", 
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    
                    # Count reps
                    if angle < 90 and direction == 0:
                        direction = 1
                    elif angle > 160 and direction == 1:
                        rep_count += 1
                        direction = 0
                    
                elif exercise == "pushup":
                    # Get right arm landmarks
                    shoulder = [landmarks.landmark[12].x, landmarks.landmark[12].y]
                    elbow = [landmarks.landmark[14].x, landmarks.landmark[14].y]
                    wrist = [landmarks.landmark[16].x, landmarks.landmark[16].y]
                    
                    angle = predictor.calculate_angle(shoulder, elbow, wrist)
                    cv2.putText(frame, f"Elbow Angle: {int(angle)}°", 
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    
                    # Count reps
                    if angle < 90 and direction == 0:
                        direction = 1
                    elif angle > 160 and direction == 1:
                        rep_count += 1
                        direction = 0
                
                # Display rep count
                cv2.putText(frame, f"Reps: {rep_count}", 
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Track exercise changes
                if exercise != last_exercise:
                    if last_exercise:
                        print(f"Finished {last_exercise}, reps: {rep_count}")
                        performance_metrics[last_exercise] = rep_count
                    rep_count = 0
                    direction = 0
                    last_exercise = exercise
        
        # Display instructions
        cv2.putText(frame, "Press 'q' to quit", 
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Exercise Recognition', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    # Print performance summary
    print("\nWorkout Summary:")
    for exercise, reps in performance_metrics.items():
        print(f"{exercise}: {reps} reps")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()