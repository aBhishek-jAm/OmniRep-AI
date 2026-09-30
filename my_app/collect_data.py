# collect_data.py
import cv2
import mediapipe as mp
import numpy as np
import csv
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Define classes
EXERCISES = ['squat', 'pushup', 'jumping_jack']
DATA_FILE = 'pose_data.csv'

def extract_landmarks(results):
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    return landmarks

cap = cv2.VideoCapture(0)
exercise = input("Enter exercise label (squat/pushup/jumping_jack): ")
samples = 0

with open(DATA_FILE, mode='a', newline='') as f:
    csv_writer = csv.writer(f)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = extract_landmarks(results)
            landmarks.append(exercise)
            csv_writer.writerow(landmarks)
            samples += 1

        cv2.putText(frame, f"Collecting {exercise}: {samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Pose Collection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q') or samples >= 200:
            break

cap.release()
cv2.destroyAllWindows()
