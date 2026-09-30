import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained model and label encoder
model = joblib.load("exercise_classifier.pkl")
le = joblib.load("label_encoder.pkl")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Rep counting variables
rep_count = 0
direction = 0  # 0 = down, 1 = up

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def extract_landmarks(results):
    """Extract normalized 3D landmarks with visibility."""
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    return landmarks

def get_coords(landmarks, index):
    """Get (x, y) coordinates of a landmark."""
    return [landmarks[index].x, landmarks[index].y]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert the frame
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image_height, image_width, _ = frame.shape

    if results.pose_landmarks:
        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        flat_landmarks = extract_landmarks(results)

        if len(flat_landmarks) == 132:
            # Predict exercise class
            probs = model.predict_proba([flat_landmarks])[0]
            pred_index = np.argmax(probs)
            exercise = le.inverse_transform([pred_index])[0]
            confidence = round(probs[pred_index] * 100, 2)

            # Display prediction
            cv2.putText(frame, f"{exercise} - {confidence}%", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Count repetitions for specific exercises
            if exercise == "squat":
                hip = get_coords(landmarks, 24)    # Right hip
                knee = get_coords(landmarks, 26)   # Right knee
                ankle = get_coords(landmarks, 28)  # Right ankle

                angle = calculate_angle(hip, knee, ankle)
                cv2.putText(frame, f"Knee Angle: {int(angle)}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

                if angle < 90 and direction == 0:
                    direction = 1
                elif angle > 160 and direction == 1:
                    rep_count += 1
                    direction = 0

            elif exercise == "pushup":
                shoulder = get_coords(landmarks, 12)  # Right shoulder
                elbow = get_coords(landmarks, 14)     # Right elbow
                wrist = get_coords(landmarks, 16)     # Right wrist

                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(frame, f"Elbow Angle: {int(angle)}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

                if angle < 90 and direction == 0:
                    direction = 1
                elif angle > 160 and direction == 1:
                    rep_count += 1
                    direction = 0

            # Display rep count
            # cv2.putText(frame, f"Reps: {rep_count}", (10, 120),
                      #  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("Live Exercise Recognition with Reps", frame)

    # Exit on 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
