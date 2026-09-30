"""
Real-time exercise prediction with rep counting.

Uses webcam feed + pre-trained ML model to classify exercises and count reps
using joint angle heuristics.

Usage:
    python -m src.prediction [--model-dir models] [--confidence 0.7]
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import argparse
import time
from collections import deque

from src.utils import calculate_angle, get_landmark_coords, extract_basic_landmarks, extract_enhanced_landmarks


# Default directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# Configuration
WINDOW_SIZE = 10        # Number of frames to average predictions over
SMOOTHING_FACTOR = 0.3  # Exponential smoothing factor


class ExercisePredictor:
    """
    Real-time exercise predictor with smoothed predictions and rep counting.

    Attributes:
        pose: MediaPipe Pose instance
        model: Trained classifier
        label_encoder: Fitted LabelEncoder
        confidence_threshold: Minimum confidence to accept a prediction
    """

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, confidence_threshold=0.7):
        """
        Initialize the predictor with a pre-trained model.

        Args:
            model_dir: Directory containing exercise_classifier.pkl and label_encoder.pkl
            confidence_threshold: Minimum confidence to display prediction
        """
        # Load model
        model_path = os.path.join(model_dir, 'exercise_classifier.pkl')
        le_path = os.path.join(model_dir, 'label_encoder.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                "Please train a model first: python -m src.training"
            )

        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(le_path)
        self.confidence_threshold = confidence_threshold

        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Prediction smoothing
        self.smoothed_probs = None
        self.current_prediction = None
        self.last_prediction_time = time.time()

        # Rep counting state
        self.rep_counts = {}
        self.rep_direction = 0  # 0 = down, 1 = up
        self.last_exercise = None

        # Determine expected feature count
        self._determine_feature_mode()

        print(f"  ✓ Model loaded from {model_dir}")
        print(f"  ✓ Classes: {list(self.label_encoder.classes_)}")
        print(f"  ✓ Feature mode: {'enhanced (138)' if self.enhanced_mode else 'basic (132)'}")

    def _determine_feature_mode(self):
        """Determine if the model expects basic (132) or enhanced (138) features."""
        try:
            # Try to infer from model's expected input
            if hasattr(self.model, 'n_features_in_'):
                n_features = self.model.n_features_in_
            elif hasattr(self.model, 'named_steps'):
                # Pipeline - check the scaler or classifier
                for step_name, step in self.model.named_steps.items():
                    if hasattr(step, 'n_features_in_'):
                        n_features = step.n_features_in_
                        break
                else:
                    n_features = 138  # Default to enhanced
            else:
                n_features = 138

            self.enhanced_mode = (n_features > 132)
            self.expected_features = n_features
        except Exception:
            self.enhanced_mode = True
            self.expected_features = 138

    def predict_frame(self, frame):
        """
        Process a video frame and predict the exercise being performed.

        Args:
            frame: BGR OpenCV image

        Returns:
            tuple: (prediction_tuple_or_None, pose_landmarks_or_None)
                   prediction_tuple = (exercise_name, confidence)
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        if results.pose_landmarks:
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

            # Extract features
            if self.enhanced_mode:
                features = extract_enhanced_landmarks(results)
            else:
                features = extract_basic_landmarks(results)

            if len(features) == self.expected_features:
                # Get prediction probabilities
                current_probs = self.model.predict_proba([features])[0]

                # Apply exponential smoothing
                if self.smoothed_probs is None:
                    self.smoothed_probs = current_probs
                else:
                    self.smoothed_probs = (
                        SMOOTHING_FACTOR * current_probs +
                        (1 - SMOOTHING_FACTOR) * self.smoothed_probs
                    )

                # Get best prediction
                pred_index = np.argmax(self.smoothed_probs)
                confidence = self.smoothed_probs[pred_index]
                exercise = self.label_encoder.inverse_transform([pred_index])[0]

                if confidence > self.confidence_threshold:
                    self.current_prediction = (exercise, confidence)
                    self.last_prediction_time = time.time()
                elif time.time() - self.last_prediction_time > 2:
                    self.current_prediction = None

                return self.current_prediction, results.pose_landmarks

        return None, (results.pose_landmarks if results.pose_landmarks else None)

    def count_reps(self, frame, exercise, landmarks):
        """
        Count repetitions for the detected exercise using joint angle heuristics.

        Args:
            frame: OpenCV image for annotation
            exercise: Detected exercise name
            landmarks: MediaPipe pose landmarks

        Returns:
            int: Current rep count for this exercise
        """
        lm = landmarks.landmark

        # Track exercise changes
        if exercise != self.last_exercise:
            if self.last_exercise and self.last_exercise in self.rep_counts:
                print(f"  Finished {self.last_exercise}: {self.rep_counts.get(self.last_exercise, 0)} reps")
            self.rep_direction = 0
            self.last_exercise = exercise

        if exercise not in self.rep_counts:
            self.rep_counts[exercise] = 0

        angle = None

        if exercise == "squat":
            hip = get_landmark_coords(lm, 24)
            knee = get_landmark_coords(lm, 26)
            ankle = get_landmark_coords(lm, 28)
            angle = calculate_angle(hip, knee, ankle)

            cv2.putText(frame, f"Knee Angle: {int(angle)}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

            if angle < 90 and self.rep_direction == 0:
                self.rep_direction = 1
            elif angle > 160 and self.rep_direction == 1:
                self.rep_counts[exercise] += 1
                self.rep_direction = 0

        elif exercise == "pushup":
            shoulder = get_landmark_coords(lm, 12)
            elbow = get_landmark_coords(lm, 14)
            wrist = get_landmark_coords(lm, 16)
            angle = calculate_angle(shoulder, elbow, wrist)

            cv2.putText(frame, f"Elbow Angle: {int(angle)}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

            if angle < 90 and self.rep_direction == 0:
                self.rep_direction = 1
            elif angle > 160 and self.rep_direction == 1:
                self.rep_counts[exercise] += 1
                self.rep_direction = 0

        elif exercise == "jumping_jack":
            # Use shoulder-hip-wrist angle for jumping jacks
            shoulder = get_landmark_coords(lm, 12)
            hip = get_landmark_coords(lm, 24)
            wrist = get_landmark_coords(lm, 16)
            angle = calculate_angle(shoulder, hip, wrist)

            if angle > 150 and self.rep_direction == 0:
                self.rep_direction = 1
            elif angle < 90 and self.rep_direction == 1:
                self.rep_counts[exercise] += 1
                self.rep_direction = 0

        elif exercise == "lunge":
            hip = get_landmark_coords(lm, 24)
            knee = get_landmark_coords(lm, 26)
            ankle = get_landmark_coords(lm, 28)
            angle = calculate_angle(hip, knee, ankle)

            cv2.putText(frame, f"Knee Angle: {int(angle)}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

            if angle < 100 and self.rep_direction == 0:
                self.rep_direction = 1
            elif angle > 150 and self.rep_direction == 1:
                self.rep_counts[exercise] += 1
                self.rep_direction = 0

        return self.rep_counts.get(exercise, 0)

    def get_workout_summary(self):
        """Get a summary of all exercise rep counts."""
        return dict(self.rep_counts)


def main():
    parser = argparse.ArgumentParser(description="Real-time exercise recognition")
    parser.add_argument('--model-dir', type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing trained model files")
    parser.add_argument('--confidence', type=float, default=0.7,
                        help="Minimum prediction confidence (default: 0.7)")
    parser.add_argument('--camera', type=int, default=0,
                        help="Camera index (default: 0)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  FitVision AI - Live Exercise Recognition")
    print("=" * 60)

    predictor = ExercisePredictor(args.model_dir, args.confidence)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("\n  Camera opened. Press 'q' to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # Predict
        prediction, landmarks = predictor.predict_frame(frame)

        if prediction:
            exercise, confidence = prediction

            # Display prediction
            color = (0, 255, 0) if confidence > 0.85 else (0, 255, 255)
            cv2.putText(frame, f"{exercise} ({confidence:.1%})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Count reps
            if landmarks:
                reps = predictor.count_reps(frame, exercise, landmarks)
                cv2.putText(frame, f"Reps: {reps}",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Instructions
        cv2.putText(frame, "Press 'q' to quit",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('FitVision AI - Exercise Recognition', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Print summary
    summary = predictor.get_workout_summary()
    if summary:
        print("\n" + "=" * 40)
        print("  Workout Summary")
        print("=" * 40)
        for ex, reps in summary.items():
            print(f"    {ex}: {reps} reps")
        print("=" * 40 + "\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
