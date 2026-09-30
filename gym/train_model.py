import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DATA_DIR = 'exercise_data'

def load_data():
    """Load all CSV files from exercise directories"""
    all_data = []
    for exercise in os.listdir(BASE_DATA_DIR):
        exercise_dir = os.path.join(BASE_DATA_DIR, exercise)
        if os.path.isdir(exercise_dir):
            for csv_file in os.listdir(exercise_dir):
                if csv_file.endswith('.csv'):
                    file_path = os.path.join(exercise_dir, csv_file)
                    df = pd.read_csv(file_path)
                    all_data.append(df)
    
    if not all_data:
        raise ValueError("No training data found! Please collect data first.")
    
    return pd.concat(all_data, ignore_index=True)

def preprocess_data(df):
    """Preprocess the data"""
    # Drop rows with missing values
    df = df.dropna()
    
    # Separate features and labels
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

def train_model(X_train, y_train):
    """Train and evaluate different models"""
    # Create a pipeline with standardization and classifier
    pipelines = {
        'random_forest': make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
        'svm': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        'knn': make_pipeline(StandardScaler(), KNeighborsClassifier())
    }
    
    # Hyperparameter grids
    param_grids = {
        'random_forest': {
            'randomforestclassifier__n_estimators': [100, 200],
            'randomforestclassifier__max_depth': [None, 10, 20]
        },
        'svm': {
            'svc__C': [0.1, 1, 10],
            'svc__kernel': ['rbf', 'poly']
        },
        'knn': {
            'kneighborsclassifier__n_neighbors': [3, 5, 7],
            'kneighborsclassifier__weights': ['uniform', 'distance']
        }
    }
    
    best_model = None
    best_score = 0
    
    for model_name in pipelines:
        print(f"\nTraining {model_name}...")
        grid_search = GridSearchCV(pipelines[model_name], 
                                 param_grids[model_name], 
                                 cv=5, 
                                 scoring='accuracy',
                                 n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best {model_name} params:", grid_search.best_params_)
        print(f"Best {model_name} score: {grid_search.best_score_:.4f}")
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
    
    print(f"\nSelected best model: {type(best_model.named_steps[list(best_model.named_steps.keys())[1]]).__name__}")
    return best_model

def evaluate_model(model, X_test, y_test, le):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Feature importance (if available)
    if hasattr(model.named_steps[list(model.named_steps.keys())[1]], 'feature_importances_'):
        importances = model.named_steps[list(model.named_steps.keys())[1]].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print top 20 features
        print("\nTop 20 Important Features:")
        for i in range(20):
            print(f"{i+1}. Feature {indices[i]} ({importances[indices[i]]:.4f})")

def save_model(model, le):
    """Save the trained model and label encoder"""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, 'models/exercise_classifier.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    print("\nModel saved to 'models/exercise_classifier.pkl'")

def main():
    # Load and preprocess data
    df = load_data()
    X, y, le = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    evaluate_model(model, X_test, y_test, le)
    
    # Save model
    save_model(model, le)

if __name__ == "__main__":
    main()