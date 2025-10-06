from src.preprocessing import load_and_preprocess
from src.model import train_model, predict_candidates
from src.evaluation import evaluate_model
from src.utils import save_report
import pandas as pd
import os

def main():
    try:
        # Create necessary directories
        os.makedirs("outputs/figures", exist_ok=True)
        os.makedirs("outputs/reports", exist_ok=True)
        
        # 1. Load data - Fixed file name
        data_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
        
        # Check if file exists
        if not os.path.exists(data_path):
            # Try alternative path
            data_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found. Please ensure the CSV file exists at: {data_path}")
        
        print(f"Loading data from: {data_path}")
        X_train, X_test, y_train, y_test, le_target, features = load_and_preprocess(data_path)
        
        # 2. Train model
        print("Training model...")
        model = train_model(X_train, y_train)
        
        # 3. Evaluate
        print("Evaluating model...")
        acc, report = evaluate_model(model, X_test, y_test, le_target)
        save_report("outputs/reports/model_summary.txt", acc, report)
        print(f"Model accuracy: {acc:.4f}")
        
        # 4. Predict on hypothetical data
        print("Making predictions on candidate data...")
        candidates = pd.DataFrame({
            "age": [25, 40, 30],
            "education_level": [2, 5, 4],
            "years_of_experience": [2, 15, 5],
            "technical_test_score": [60, 100, 80],
            "interview_score": [3, 4, 2],
            "previous_employment_enc": [1, 1, 0]
        })
        
        # Ensure candidate data has the same features as training data
        preds = predict_candidates(model, candidates)
        candidates["Predicted Suitability"] = [le_target.classes_[p] for p in preds]
        
        print("\nCandidate Predictions:")
        print(candidates)
        
        # Save predictions
        candidates.to_csv("outputs/reports/candidate_predictions.csv", index=False)
        print("\nPredictions saved to: outputs/reports/candidate_predictions.csv")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please ensure:")
        print("1. The data file exists in the correct location")
        print("2. All required packages are installed (pip install -r requirements.txt)")
        print("3. The directory structure is correct")

if __name__ == "__main__":
    main()