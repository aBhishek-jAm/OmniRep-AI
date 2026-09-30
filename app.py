"""
FitVision AI - Streamlit Web App

A web-based AI personal fitness trainer that uses your webcam to
recognize exercises and count repetitions in real-time.

Usage:
    streamlit run app.py
    or
    python main.py webapp
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import calculate_angle, get_mediapipe_coordinates, draw_angle_on_image, draw_probability_bars

# Try importing webrtc components
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Try importing TensorFlow for LSTM model
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Flatten,
                                         Bidirectional, Permute, multiply)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Try importing sklearn model
try:
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================
# LSTM Attention Model (from Exercise_Recognition_AI)
# ============================================================

def attention_block(inputs, time_steps):
    """Attention layer for deep neural network."""
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def build_lstm_model(hidden_units=256, sequence_length=30, num_input_values=33*4, num_classes=3):
    """Build LSTM + Attention model and load pre-trained weights."""
    if not TENSORFLOW_AVAILABLE:
        return None

    inputs = Input(shape=(sequence_length, num_input_values))
    lstm_out = Bidirectional(LSTM(hidden_units, return_sequences=True))(inputs)
    attention_mul = attention_block(lstm_out, sequence_length)
    attention_mul = Flatten()(attention_mul)
    x = Dense(2 * hidden_units, activation='relu')(attention_mul)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=x)

    # Try to load weights
    weight_paths = [
        os.path.join("Exercise_Recognition_AI", "models", "LSTM_Attention.h5"),
        os.path.join("models", "LSTM_Attention.h5"),
    ]
    for path in weight_paths:
        if os.path.exists(path):
            model.load_weights(path)
            return model

    return None


# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(
    page_title="FitVision AI - Personal Fitness Trainer",
    page_icon="🏋️",
    layout="wide"
)

st.write("# 🏋️ FitVision AI - Personal Fitness Trainer")
st.markdown("Real-time exercise recognition and rep counting powered by AI & Computer Vision")

# Sidebar - Settings
st.sidebar.write("## ⚙️ Settings")

# Model selection
model_choice = st.sidebar.selectbox(
    "Model Type",
    ["ML Classifier (sklearn)", "LSTM + Attention (TensorFlow)"],
    index=0
)

threshold1 = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.5)
threshold2 = st.sidebar.slider("Min Tracking Confidence", 0.0, 1.0, 0.5)
threshold3 = st.sidebar.slider("Min Classification Confidence", 0.0, 1.0, 0.5)

# Load selected model
loaded_model = None
label_encoder = None
model_type = None

if model_choice == "ML Classifier (sklearn)" and SKLEARN_AVAILABLE:
    model_paths = [
        os.path.join("models", "exercise_classifier.pkl"),
        os.path.join("my_app", "exercise_classifier.pkl"),
    ]
    le_paths = [
        os.path.join("models", "label_encoder.pkl"),
        os.path.join("my_app", "label_encoder.pkl"),
    ]
    for mp_, lp_ in zip(model_paths, le_paths):
        if os.path.exists(mp_) and os.path.exists(lp_):
            loaded_model = joblib.load(mp_)
            label_encoder = joblib.load(lp_)
            model_type = "sklearn"
            st.sidebar.success(f"✓ Loaded sklearn model from {mp_}")
            break
    if loaded_model is None:
        st.sidebar.warning("⚠ No sklearn model found. Train one with: `python main.py train`")

elif model_choice == "LSTM + Attention (TensorFlow)" and TENSORFLOW_AVAILABLE:
    loaded_model = build_lstm_model()
    if loaded_model:
        model_type = "lstm"
        st.sidebar.success("✓ Loaded LSTM + Attention model")
    else:
        st.sidebar.warning("⚠ LSTM weights not found")

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Main content
st.write("## 🎥 Activate the AI")

if not WEBRTC_AVAILABLE:
    st.error("⚠ streamlit-webrtc not installed. Install with: `pip install streamlit-webrtc`")
    st.info("As an alternative, use the CLI: `python main.py predict`")
else:
    pose = mp_pose.Pose(min_detection_confidence=threshold1, min_tracking_confidence=threshold2)

    class VideoProcessor:
        """Processes video frames for exercise recognition."""

        def __init__(self):
            if model_type == "lstm":
                self.actions = np.array(['curl', 'press', 'squat'])
                self.sequence_length = 30
            elif model_type == "sklearn" and label_encoder is not None:
                self.actions = label_encoder.classes_
                self.sequence_length = 1
            else:
                self.actions = np.array(['unknown'])
                self.sequence_length = 1

            self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245),
                          (245, 16, 117), (16, 245, 245)]
            self.threshold = threshold3
            self.sequence = []
            self.current_action = ''

            # Rep counters
            self.rep_counts = {a: 0 for a in self.actions}
            self.rep_stages = {a: None for a in self.actions}

        def process(self, image):
            """Process a video frame."""
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            if loaded_model is None:
                cv2.putText(image, "No model loaded", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return image

            # Extract keypoints
            if results.pose_landmarks:
                keypoints = np.array([[res.x, res.y, res.z, res.visibility]
                                     for res in results.pose_landmarks.landmark]).flatten()
            else:
                keypoints = np.zeros(33 * 4)

            if model_type == "lstm":
                self.sequence.append(keypoints.astype('float32'))
                self.sequence = self.sequence[-self.sequence_length:]

                if len(self.sequence) == self.sequence_length:
                    res = loaded_model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                    self.current_action = self.actions[np.argmax(res)]
                    confidence = np.max(res)

                    if confidence < self.threshold:
                        self.current_action = ''

                    image = draw_probability_bars(image, res, list(self.actions), self.colors)

            elif model_type == "sklearn":
                features = keypoints.tolist()
                if len(features) == 132:
                    try:
                        res = loaded_model.predict_proba([features])[0]
                        pred_idx = np.argmax(res)
                        self.current_action = self.actions[pred_idx]
                        confidence = res[pred_idx]

                        if confidence < self.threshold:
                            self.current_action = ''

                        image = draw_probability_bars(image, res, list(self.actions), self.colors)
                    except Exception:
                        pass

            # Rep counting
            if results.pose_landmarks and self.current_action:
                try:
                    landmarks = results.pose_landmarks.landmark
                    self._count_reps(image, landmarks, self.current_action)
                except Exception:
                    pass

            # Display action and rep counts
            if self.current_action:
                cv2.rectangle(image, (0, 0), (640, 40), (50, 50, 50), -1)
                x_offset = 3
                for action in self.actions:
                    count = self.rep_counts.get(action, 0)
                    cv2.putText(image, f'{action}: {count}', (x_offset, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    x_offset += 180

            return image

        def _count_reps(self, image, landmarks, action):
            """Count reps based on joint angles."""
            if action in ('squat',):
                shoulder = get_mediapipe_coordinates(landmarks, mp_pose, 'left', 'shoulder')
                hip = get_mediapipe_coordinates(landmarks, mp_pose, 'left', 'hip')
                knee = get_mediapipe_coordinates(landmarks, mp_pose, 'left', 'knee')
                ankle = get_mediapipe_coordinates(landmarks, mp_pose, 'left', 'ankle')
                angle = calculate_angle(hip, knee, ankle)

                if angle < 90:
                    self.rep_stages[action] = "down"
                if angle > 160 and self.rep_stages.get(action) == "down":
                    self.rep_stages[action] = "up"
                    self.rep_counts[action] = self.rep_counts.get(action, 0) + 1

                draw_angle_on_image(image, angle, knee)

            elif action in ('curl',):
                shoulder = get_mediapipe_coordinates(landmarks, mp_pose, 'left', 'shoulder')
                elbow = get_mediapipe_coordinates(landmarks, mp_pose, 'left', 'elbow')
                wrist = get_mediapipe_coordinates(landmarks, mp_pose, 'left', 'wrist')
                angle = calculate_angle(shoulder, elbow, wrist)

                if angle < 30:
                    self.rep_stages[action] = "up"
                if angle > 140 and self.rep_stages.get(action) == "up":
                    self.rep_stages[action] = "down"
                    self.rep_counts[action] = self.rep_counts.get(action, 0) + 1

                draw_angle_on_image(image, angle, elbow)

            elif action in ('press',):
                shoulder = get_mediapipe_coordinates(landmarks, mp_pose, 'left', 'shoulder')
                elbow = get_mediapipe_coordinates(landmarks, mp_pose, 'left', 'elbow')
                wrist = get_mediapipe_coordinates(landmarks, mp_pose, 'left', 'wrist')
                angle = calculate_angle(shoulder, elbow, wrist)
                shoulder2elbow_dist = abs(math.dist(shoulder, elbow))
                shoulder2wrist_dist = abs(math.dist(shoulder, wrist))

                if angle > 130 and shoulder2elbow_dist < shoulder2wrist_dist:
                    self.rep_stages[action] = "up"
                if (angle < 50 and shoulder2elbow_dist > shoulder2wrist_dist
                        and self.rep_stages.get(action) == "up"):
                    self.rep_stages[action] = "down"
                    self.rep_counts[action] = self.rep_counts.get(action, 0) + 1

                draw_angle_on_image(image, angle, elbow)

            elif action in ('pushup',):
                shoulder = get_mediapipe_coordinates(landmarks, mp_pose, 'right', 'shoulder')
                elbow = get_mediapipe_coordinates(landmarks, mp_pose, 'right', 'elbow')
                wrist = get_mediapipe_coordinates(landmarks, mp_pose, 'right', 'wrist')
                angle = calculate_angle(shoulder, elbow, wrist)

                if angle < 90:
                    self.rep_stages[action] = "down"
                if angle > 160 and self.rep_stages.get(action) == "down":
                    self.rep_stages[action] = "up"
                    self.rep_counts[action] = self.rep_counts.get(action, 0) + 1

                draw_angle_on_image(image, angle, elbow)

        def recv(self, frame):
            """Receive and process video stream from webcam."""
            img = frame.to_ndarray(format="bgr24")
            img = self.process(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # WebRTC Streamer
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_ctx = webrtc_streamer(
        key="fitvision-ai",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

# Instructions
st.write("## 📖 How to Use")
st.markdown("""
1. **Allow camera access** when prompted
2. **Position yourself** so your full body is visible
3. **Perform exercises** - the AI will recognize them and count reps
4. **Adjust settings** in the sidebar for better accuracy
""")

st.write("## 🛠️ CLI Commands")
st.code("""
# Collect training data
python main.py collect --exercise squat --samples 300

# Train the model
python main.py train

# Real-time prediction (OpenCV window)
python main.py predict

# Track specific exercise with angles
python main.py track --type pushup --source 0
""", language="bash")
