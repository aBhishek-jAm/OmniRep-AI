# FitVision AI 🏋️‍♂️🤖

**AI-Powered Exercise Recognition & Rep Counting System**

A unified computer vision project that uses MediaPipe pose estimation and machine learning
to recognize exercises in real-time, count repetitions, and provide form feedback.

## 🎯 Features

- **Real-time Exercise Recognition** — Classifies exercises from webcam using ML models
- **Automatic Rep Counting** — Counts reps using joint angle heuristics
- **Form Tracking** — Visual angle overlays and progress bars (GymLytics-style)
- **Multiple Exercises** — Squat, Pushup, Lunge, Plank, Shoulder Tap, Jumping Jack, and more
- **Two Model Options** — sklearn Random Forest/SVM/KNN or LSTM + Attention (TensorFlow)
- **Web App** — Streamlit-based web interface with WebRTC camera access
- **Data Pipeline** — Built-in data collection and model training tools

## 📁 Project Structure

```
exercise/
├── main.py                    # CLI entry point (all commands)
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── src/
│   ├── __init__.py
│   ├── utils.py               # Shared utilities (angles, landmarks, visualization)
│   ├── data_collection.py     # Collect training data from webcam
│   ├── training.py            # Train ML models with auto-selection
│   ├── prediction.py          # Real-time prediction & rep counting
│   └── exercises/             # Rule-based exercise trackers
│       ├── __init__.py
│       ├── base.py            # Abstract Exercise base class
│       ├── threaded_camera.py # Threaded camera capture
│       ├── squat.py           # Squat tracker with angle visualization
│       ├── pushup.py          # Pushup tracker
│       ├── lunges.py          # Lunge tracker
│       ├── plank.py           # Plank hold timer
│       └── shoulder_tap.py    # Shoulder tap tracker
├── data/
│   └── exercise_data/         # Training data (CSV files per exercise)
├── models/                    # Trained model files (.pkl, .h5)
└── venv/                      # Python virtual environment
```

## 🚀 Quick Start

### 1. Setup

```bash
# Activate virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Collect Training Data

```bash
# Collect 300 samples of squats
python main.py collect --exercise squat --samples 300

# Collect pushup data
python main.py collect --exercise pushup --samples 300
```

### 3. Train the Model

```bash
python main.py train
```

This will:
- Load all CSV data from `data/exercise_data/`
- Train Random Forest, SVM, and KNN models
- Automatically select the best model
- Save to `models/exercise_classifier.pkl`

### 4. Run Real-Time Prediction

```bash
# OpenCV window (recommended)
python main.py predict

# With custom confidence threshold
python main.py predict --confidence 0.8
```

### 5. Track Specific Exercises (with angle visualization)

```bash
python main.py track --type squat --source 0      # Squat with angles
python main.py track --type pushup --source 0      # Pushup form tracking
python main.py track --type plank --source 0       # Plank hold timer
python main.py track --type lunges --source 0      # Lunge tracking
python main.py track --type shouldertap --source 0 # Shoulder tap counter
```

### 6. Launch Web App

```bash
python main.py webapp
# or
streamlit run app.py
```

## 🛠️ CLI Reference

```
python main.py <command> [options]

Commands:
  collect    Collect training data from webcam
  train      Train the ML classification model
  predict    Real-time exercise recognition with rep counting
  track      Rule-based exercise tracking with angle visualization
  webapp     Launch the Streamlit web application

Use 'python main.py <command> --help' for command-specific options.
```

## 📊 Supported Exercises

| Exercise      | ML Recognition | Rule-Based Tracking | Rep Counting |
|---------------|:--------------:|:-------------------:|:------------:|
| Squat         | ✅             | ✅                  | ✅           |
| Pushup        | ✅             | ✅                  | ✅           |
| Jumping Jack  | ✅             | —                   | ✅           |
| Lunge         | ✅             | ✅                  | ✅           |
| Plank         | ✅             | ✅                  | ⏱ (timer)   |
| Shoulder Tap  | —              | ✅                  | ✅           |
| Curl          | ✅ (LSTM)      | —                   | ✅           |
| Press         | ✅ (LSTM)      | —                   | ✅           |

## 🔧 Technology Stack

- **Pose Estimation**: MediaPipe BlazePose (33 body landmarks)
- **ML Models**: scikit-learn (Random Forest, SVM, KNN)
- **Deep Learning**: TensorFlow/Keras (Bi-LSTM + Attention)
- **Computer Vision**: OpenCV
- **Web App**: Streamlit + WebRTC
- **Python**: 3.12+

## 📝 Credits

This project consolidates and improves code from multiple sources:
- [Exercise_Recognition_AI](https://github.com/chrisprasanna/Exercise_Recognition_AI) — LSTM + Attention model
- [GymLytics](https://github.com/) — Rule-based exercise tracking with angle visualization
- Custom ML pipeline for data collection, training, and prediction

## 📄 License

MIT License
