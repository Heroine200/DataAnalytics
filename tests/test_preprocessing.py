import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.preprocessing import load_and_preprocess_data, encode_categorical_features, prepare_features_target, split_data

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        # Create sample data for testing
        self.sample_data = {
            'Age': [25, 30, 35],
            'Education': [3, 4, 2],
            'TotalWorkingYears': [2, 5, 10],
            'JobLevel': [1, 2, 3],
            'PerformanceRating': [3, 4, 3],
            'NumCompaniesWorked': [1, 0, 2],
            'Attrition': ['No', 'Yes', 'No']
        }
        self.df_raw = pd.DataFrame(self.sample_data)
        
    def test_encode_categorical_features(self):
        # Test encoding functionality
        df = pd.DataFrame({
            'previous_employment': ['Yes', 'No', 'Yes'],
            'suitable_for_employment': ['Yes', 'No', 'Yes'],
            'age': [25, 30, 35],
            'education_level': [3, 4, 2],
            'years_of_experience': [2, 5, 10],
            'technical_test_score': [1.0, 2.0, 3.0],
            'interview_score': [3.0, 4.0, 3.0]
        })
        
        df_encoded, le_prev, le_target = encode_categorical_features(df)
        
        self.assertIn('previous_employment_encoded', df_encoded.columns)
        self.assertIn('suitable_for_employment_encoded', df_encoded.columns)
        
        # Check that encoding produced integers
        self.assertTrue(np.issubdtype(df_encoded['previous_employment_encoded'].dtype, np.integer))
        self.assertTrue(np.issubdtype(df_encoded['suitable_for_employment_encoded'].dtype, np.integer))
        
        # Check that values are 0 and 1
        unique_prev = set(df_encoded['previous_employment_encoded'].unique())
        unique_target = set(df_encoded['suitable_for_employment_encoded'].unique())
        self.assertEqual(unique_prev, {0, 1})
        self.assertEqual(unique_target, {0, 1})

    def test_prepare_features_target(self):
        """Test feature and target preparation"""
        # Create test data with encoded columns
        df_encoded = pd.DataFrame({
            'age': [25, 30, 35],
            'education_level': [3, 4, 2],
            'years_of_experience': [2, 5, 10],
            'technical_test_score': [1.0, 2.0, 3.0],
            'interview_score': [3.0, 4.0, 3.0],
            'previous_employment_encoded': [1, 0, 1],
            'suitable_for_employment_encoded': [1, 0, 1]
        })
        
        X, y = prepare_features_target(df_encoded)
        
        # Check shapes and types
        self.assertEqual(X.shape[1], 6)  # 6 features
        self.assertEqual(len(y), 3)      # 3 target values
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_split_data(self):
        """Test data splitting functionality"""
        # Create test data
        X = pd.DataFrame({
            'age': [25, 30, 35, 40, 45, 50],
            'education_level': [3, 4, 2, 4, 3, 2],
            'years_of_experience': [2, 5, 10, 8, 15, 12],
            'technical_test_score': [2.0, 3.0, 1.0, 3.0, 2.0, 1.0],
            'interview_score': [3.0, 4.0, 3.0, 4.0, 3.0, 3.0],
            'previous_employment_encoded': [1, 0, 1, 1, 0, 1]
        })
        y = pd.Series([1, 1, 0, 1, 0, 0])
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.33, random_state=42)
        
        # Check split sizes (33% of 6 samples ≈ 2 test samples)
        self.assertEqual(len(X_train), 4)
        self.assertEqual(len(X_test), 2)
        self.assertEqual(len(y_train), 4)
        self.assertEqual(len(y_test), 2)

if __name__ == '__main__':
    unittest.main()