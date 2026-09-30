"""
Data collection module for gathering pose landmark training data.

Captures webcam footage, extracts MediaPipe pose landmarks, and saves them
as CSV files organized by exercise type. Supports both basic (132 features)
and enhanced (138 features with distances and angles) landmark extraction.

Usage:
    python -m src.data_collection [--exercise squat] [--samples 300] [--enhanced]
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import argparse
from datetime import datetime

from src.utils import extract_basic_landmarks, extract_enhanced_landmarks


# Supported exercise types
EXERCISES = ['squat', 'pushup', 'jumping_jack', 'rest', 'lunge', 'plank']
BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'exercise_data')


def create_directories():
    """Create directories for each exercise type if they don't exist."""
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    for exercise in EXERCISES:
        os.makedirs(os.path.join(BASE_DATA_DIR, exercise), exist_ok=True)


def get_csv_header(enhanced=True):
    """
    Generate CSV header row.

    Args:
        enhanced: If True, include distance and angle feature headers

    Returns:
        list: Column header names
    """
    header = []
    for i in range(33):  # 33 landmarks
        header.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"])

    if enhanced:
        header.extend([
            "shoulder_width", "torso_length",
            "left_elbow_angle", "right_elbow_angle",
            "left_knee_angle", "right_knee_angle"
        ])

    header.append("label")
    return header


def collect_data(exercise_name, samples_per_session=300, enhanced=True):
    """
    Collect pose landmark data for a specific exercise.

    Opens the webcam, processes frames with MediaPipe Pose, extracts landmarks,
    and writes them to a CSV file.

    Args:
        exercise_name: Name of the exercise to collect data for
        samples_per_session: Number of samples to collect (default: 300)
        enhanced: Whether to include engineered features (default: True)
    """
    create_directories()

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    samples = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = os.path.join(BASE_DATA_DIR, exercise_name, f"{exercise_name}_{timestamp}.csv")

    extract_fn = extract_enhanced_landmarks if enhanced else extract_basic_landmarks

    print(f"\n{'='*50}")
    print(f"  Collecting data for: {exercise_name.upper()}")
    print(f"  Target samples: {samples_per_session}")
    print(f"  Enhanced features: {enhanced}")
    print(f"  Output: {data_file}")
    print(f"{'='*50}")
    print("  Press 'q' to stop early\n")

    with open(data_file, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(get_csv_header(enhanced))

        while samples < samples_per_session:
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = extract_fn(results)

                if landmarks:
                    landmarks.append(exercise_name)
                    csv_writer.writerow(landmarks)
                    samples += 1

                    # Display info
                    cv2.putText(frame, f"Collecting {exercise_name}: {samples}/{samples_per_session}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Progress bar
                    progress = int((samples / samples_per_session) * frame.shape[1])
                    cv2.rectangle(frame, (0, frame.shape[0] - 20),
                                  (progress, frame.shape[0]), (0, 255, 0), -1)

            cv2.putText(frame, "Press 'q' to quit",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("FitVision - Data Collection", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print(f"\nCollected {samples} samples of '{exercise_name}' → {data_file}")


def main():
    parser = argparse.ArgumentParser(description="Collect exercise pose data for training")
    parser.add_argument('--exercise', '-e', type=str, default=None,
                        help=f"Exercise type: {EXERCISES}")
    parser.add_argument('--samples', '-s', type=int, default=300,
                        help="Number of samples per session (default: 300)")
    parser.add_argument('--enhanced', action='store_true', default=True,
                        help="Include distance and angle features (default: True)")
    parser.add_argument('--basic', action='store_true',
                        help="Use basic landmarks only (132 features)")
    args = parser.parse_args()

    exercise = args.exercise
    if exercise is None:
        print(f"\nAvailable exercises: {EXERCISES}")
        exercise = input("Enter exercise label: ").lower().strip()

    while exercise not in EXERCISES:
        print(f"Invalid! Choose from: {EXERCISES}")
        exercise = input("Enter exercise label: ").lower().strip()

    enhanced = not args.basic
    collect_data(exercise, args.samples, enhanced)


if __name__ == "__main__":
    main()
