from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

def train_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    """
    Train a Decision Tree classifier
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model

def predict_employment(model, candidate_data, feature_names):
    """
    Predict employment suitability for a candidate
    """
    # Ensure candidate_data is in correct format
    if isinstance(candidate_data, dict):
        candidate_df = pd.DataFrame([candidate_data])
        # Reorder columns to match training data
        candidate_df = candidate_df[feature_names]
    else:
        candidate_df = candidate_data
    
    prediction = model.predict(candidate_df)
    probability = model.predict_proba(candidate_df)
    
    return prediction, probability

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'predictions': y_pred
    }