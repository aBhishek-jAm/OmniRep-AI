"""
Model training module with automated model selection.

Loads CSV data from exercise_data directories, trains multiple ML models
(Random Forest, SVM, KNN) with hyperparameter tuning, and saves the best model.

Usage:
    python -m src.training [--data-dir data/exercise_data] [--output-dir models]
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# Default directories (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'exercise_data')
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'models')


def load_data(data_dir):
    """
    Load all CSV files from exercise subdirectories.

    Args:
        data_dir: Root directory containing exercise subdirectories with CSV files

    Returns:
        pd.DataFrame: Combined data from all CSV files

    Raises:
        ValueError: If no training data is found
    """
    all_data = []
    for exercise in os.listdir(data_dir):
        exercise_dir = os.path.join(data_dir, exercise)
        if os.path.isdir(exercise_dir):
            for csv_file in os.listdir(exercise_dir):
                if csv_file.endswith('.csv'):
                    file_path = os.path.join(exercise_dir, csv_file)
                    try:
                        df = pd.read_csv(file_path)
                        all_data.append(df)
                        print(f"  Loaded: {file_path} ({len(df)} samples)")
                    except Exception as e:
                        print(f"  Warning: Could not load {file_path}: {e}")

    if not all_data:
        raise ValueError(
            f"No training data found in '{data_dir}'!\n"
            "Please collect data first using: python -m src.data_collection"
        )

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n  Total samples loaded: {len(combined)}")
    return combined


def preprocess_data(df):
    """
    Preprocess the data: drop NaN rows, separate features/labels, encode labels.

    Args:
        df: Raw DataFrame with last column as label

    Returns:
        tuple: (X features, y encoded labels, LabelEncoder)
    """
    df = df.dropna()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\n  Classes found: {list(le.classes_)}")
    print(f"  Feature count: {X.shape[1]}")
    for cls in le.classes_:
        print(f"    {cls}: {(y == cls).sum()} samples")

    return X, y_encoded, le


def train_models(X_train, y_train):
    """
    Train multiple ML models with hyperparameter tuning and return the best one.

    Models trained:
    - Random Forest
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Best model pipeline (with StandardScaler + classifier)
    """
    pipelines = {
        'Random Forest': make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
        'SVM': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier()),
    }

    param_grids = {
        'Random Forest': {
            'randomforestclassifier__n_estimators': [100, 200],
            'randomforestclassifier__max_depth': [None, 10, 20],
        },
        'SVM': {
            'svc__C': [0.1, 1, 10],
            'svc__kernel': ['rbf', 'poly'],
        },
        'KNN': {
            'kneighborsclassifier__n_neighbors': [3, 5, 7],
            'kneighborsclassifier__weights': ['uniform', 'distance'],
        },
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name in pipelines:
        print(f"\n  Training {name}...")
        grid_search = GridSearchCV(
            pipelines[name],
            param_grids[name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)

        print(f"    Best params: {grid_search.best_params_}")
        print(f"    Best CV score: {grid_search.best_score_:.4f}")

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_name = name

    print(f"\n  ✓ Selected best model: {best_name} (CV accuracy: {best_score:.4f})")
    return best_model


def evaluate_model(model, X_test, y_test, le, output_dir):
    """
    Evaluate model performance and save confusion matrix plot.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        le: LabelEncoder
        output_dir: Directory to save plots
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {plot_path}")

    # Feature importance (if available)
    steps = list(model.named_steps.keys())
    classifier = model.named_steps[steps[-1]]
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\n  Top 20 Important Features:")
        for i in range(min(20, len(indices))):
            print(f"    {i+1}. Feature {indices[i]} (importance: {importances[indices[i]]:.4f})")


def save_model(model, le, output_dir):
    """
    Save the trained model and label encoder.

    Args:
        model: Trained model
        le: LabelEncoder
        output_dir: Directory to save model files
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, 'exercise_classifier.pkl')
    le_path = os.path.join(output_dir, 'label_encoder.pkl')

    joblib.dump(model, model_path)
    joblib.dump(le, le_path)

    print(f"\n  ✓ Model saved: {model_path}")
    print(f"  ✓ Label encoder saved: {le_path}")


def main():
    parser = argparse.ArgumentParser(description="Train exercise classification model")
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f"Training data directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Model output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument('--test-size', type=float, default=0.2,
                        help="Test split ratio (default: 0.2)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  FitVision AI - Model Training")
    print("=" * 60)

    # Load data
    print("\n📂 Loading data...")
    df = load_data(args.data_dir)

    # Preprocess
    print("\n🔧 Preprocessing...")
    X, y, le = preprocess_data(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # Train
    print("\n🏋️ Training models...")
    model = train_models(X_train, y_train)

    # Evaluate
    print("\n📊 Evaluating...")
    evaluate_model(model, X_test, y_test, le, args.output_dir)

    # Save
    print("\n💾 Saving model...")
    save_model(model, le, args.output_dir)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
