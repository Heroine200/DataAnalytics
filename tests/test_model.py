import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.model import train_decision_tree, evaluate_model

class TestModel(unittest.TestCase):
    
    def setUp(self):
        # Create sample data for testing
        np.random.seed(42)
        self.X_train = np.random.randn(100, 6)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.randn(20, 6)
        self.y_test = np.random.randint(0, 2, 20)
    
    def test_train_decision_tree(self):
        model = train_decision_tree(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        
    def test_evaluate_model(self):
        model = train_decision_tree(self.X_train, self.y_train)
        results = evaluate_model(model, self.X_test, self.y_test)
        
        self.assertIn('accuracy', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('classification_report', results)

if __name__ == '__main__':
    unittest.main()