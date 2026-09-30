import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from datetime import datetime

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


EXERCISES = ['squat', 'pushup', 'jumping_jack', 'rest']
BASE_DATA_DIR = 'exercise_data'

def create_directories():
    """Create directories for each exercise if they don't exist"""
    if not os.path.exists(BASE_DATA_DIR):
        os.makedirs(BASE_DATA_DIR)
    for exercise in EXERCISES:
        exercise_dir = os.path.join(BASE_DATA_DIR, exercise)
        if not os.path.exists(exercise_dir):
            os.makedirs(exercise_dir)

def get_timestamp():
    """Return current timestamp for unique filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_landmarks(results):
    """Extract pose landmarks with visibility and additional features"""
    landmarks = []
    if results.pose_landmarks:
        # Basic landmarks
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        
        if len(results.pose_landmarks.landmark) >= 32:  
           
            landmarks.extend(calculate_key_distances(results.pose_landmarks.landmark))
            
            
            landmarks.extend(calculate_key_angles(results.pose_landmarks.landmark))
    return landmarks

def calculate_key_distances(landmarks):
    """Calculate normalized distances between important body points"""
    distances = []
    # Shoulder width
    left_shoulder = [landmarks[11].x, landmarks[11].y]
    right_shoulder = [landmarks[12].x, landmarks[12].y]
    shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
    distances.append(shoulder_width)
    
    # Torso length (shoulder to hip)
    torso_length = np.linalg.norm(np.array([landmarks[11].x, landmarks[11].y]) - 
                                 np.array([landmarks[23].x, landmarks[23].y]))
    distances.append(torso_length)
    
    return distances

def calculate_key_angles(landmarks):
    """Calculate important joint angles"""
    angles = []
    
    # Left elbow angle
    left_shoulder = [landmarks[11].x, landmarks[11].y]
    left_elbow = [landmarks[13].x, landmarks[13].y]
    left_wrist = [landmarks[15].x, landmarks[15].y]
    angles.append(calculate_angle(left_shoulder, left_elbow, left_wrist))
    
    # Right elbow angle
    right_shoulder = [landmarks[12].x, landmarks[12].y]
    right_elbow = [landmarks[14].x, landmarks[14].y]
    right_wrist = [landmarks[16].x, landmarks[16].y]
    angles.append(calculate_angle(right_shoulder, right_elbow, right_wrist))
    
    # Left knee angle
    left_hip = [landmarks[23].x, landmarks[23].y]
    left_knee = [landmarks[25].x, landmarks[25].y]
    left_ankle = [landmarks[27].x, landmarks[27].y]
    angles.append(calculate_angle(left_hip, left_knee, left_ankle))
    
    # Right knee angle
    right_hip = [landmarks[24].x, landmarks[24].y]
    right_knee = [landmarks[26].x, landmarks[26].y]
    right_ankle = [landmarks[28].x, landmarks[28].y]
    angles.append(calculate_angle(right_hip, right_knee, right_ankle))
    
    return angles

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def collect_data():
    create_directories()
    cap = cv2.VideoCapture(0)
    
    print("Available exercises:", EXERCISES)
    exercise = input("Enter exercise label: ").lower()
    while exercise not in EXERCISES:
        print("Invalid exercise! Please choose from:", EXERCISES)
        exercise = input("Enter exercise label: ").lower()
    
    samples_per_session = 300  # Increased samples for better variability
    samples = 0
    
    # Create a new CSV file for this session
    timestamp = get_timestamp()
    data_file = os.path.join(BASE_DATA_DIR, exercise, f"{exercise}_{timestamp}.csv")
    
    with open(data_file, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        # Write header
        header = []
        for i in range(33):  # 33 landmarks
            header.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"])
        header.extend(["shoulder_width", "torso_length",
                      "left_elbow_angle", "right_elbow_angle",
                      "left_knee_angle", "right_knee_angle",
                      "label"])
        csv_writer.writerow(header)
        
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
                landmarks = extract_landmarks(results)
                
                if landmarks:  # Only write if we got valid landmarks
                    landmarks.append(exercise)
                    csv_writer.writerow(landmarks)
                    samples += 1

                    # Display info
                    cv2.putText(frame, f"Collecting {exercise}: {samples}/{samples_per_session}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Press 'q' to quit", 
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Visual feedback - progress bar
                    progress = int((samples/samples_per_session) * frame.shape[1])
                    cv2.rectangle(frame, (0, frame.shape[0]-20), (progress, frame.shape[0]), (0, 255, 0), -1)

            cv2.imshow("Pose Collection", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {samples} samples of {exercise} in {data_file}")

if __name__ == "__main__":
    collect_data()