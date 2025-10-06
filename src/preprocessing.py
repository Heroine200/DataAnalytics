import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(data_path):
    """
    Load the dataset and perform initial preprocessing
    """
    # Load dataset
    df_raw = pd.read_csv(data_path)
    
    # Map to required features
    df = pd.DataFrame({
        "age": df_raw["Age"],
        "education_level": df_raw["Education"],
        "years_of_experience": df_raw["TotalWorkingYears"],
        "technical_test_score": df_raw["JobLevel"].astype(float),
        "interview_score": df_raw["PerformanceRating"].astype(float),
        "previous_employment": df_raw["NumCompaniesWorked"].apply(lambda x: "Yes" if pd.notnull(x) and x > 0 else "No"),
        "suitable_for_employment": df_raw["Attrition"].apply(lambda x: "No" if x == "Yes" else "Yes")
    })
    
    return df

def encode_categorical_features(df):
    """
    Encode categorical variables using LabelEncoder
    """
    df_encoded = df.copy()
    le_prev_emp = LabelEncoder()
    le_target = LabelEncoder()
    
    df_encoded['previous_employment_encoded'] = le_prev_emp.fit_transform(df_encoded['previous_employment'])
    df_encoded['suitable_for_employment_encoded'] = le_target.fit_transform(df_encoded['suitable_for_employment'])
    
    return df_encoded, le_prev_emp, le_target

def prepare_features_target(df_encoded):
    """
    Prepare features (X) and target (y) for model training
    """
    feature_columns = ['age', 'education_level', 'years_of_experience', 
                      'technical_test_score', 'interview_score', 'previous_employment_encoded']
    
    X = df_encoded[feature_columns]
    y = df_encoded['suitable_for_employment_encoded']
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
    