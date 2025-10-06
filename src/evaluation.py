import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import numpy as np
import pandas as pd

def plot_decision_tree(model, feature_names, class_names, figsize=(20, 10)):
    """
    Plot the decision tree
    """
    plt.figure(figsize=figsize)
    plot_tree(model, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title("Decision Tree for Employment Prediction")
    plt.tight_layout()
    return plt

def plot_feature_importance(model, feature_names, figsize=(10, 6)):
    """
    Plot feature importance
    """
    importance = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=figsize)
    plt.barh(feature_imp_df['feature'], feature_imp_df['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Decision Tree Feature Importance')
    plt.tight_layout()
    return plt

def plot_confusion_matrix(conf_matrix, class_names, figsize=(8, 6)):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    return plt

def create_hypothetical_candidates():
    """
    Create hypothetical candidate data for prediction examples
    """
    candidates = [
        {
            'age': 25,
            'education_level': 3,
            'years_of_experience': 2,
            'technical_test_score': 2.0,
            'interview_score': 3.0,
            'previous_employment_encoded': 1  # Yes
        },
        {
            'age': 35,
            'education_level': 4,
            'years_of_experience': 8,
            'technical_test_score': 3.0,
            'interview_score': 4.0,
            'previous_employment_encoded': 1  # Yes
        },
        {
            'age': 45,
            'education_level': 2,
            'years_of_experience': 15,
            'technical_test_score': 1.0,
            'interview_score': 3.0,
            'previous_employment_encoded': 0  # No
        }
    ]
    return candidates