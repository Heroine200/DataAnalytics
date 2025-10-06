import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.evaluation import create_hypothetical_candidates

class TestEvaluation(unittest.TestCase):
    
    def test_create_hypothetical_candidates(self):
        candidates = create_hypothetical_candidates()
        
        self.assertEqual(len(candidates), 3)
        self.assertIn('age', candidates[0])
        self.assertIn('education_level', candidates[0])
        self.assertIn('previous_employment_encoded', candidates[0])

if __name__ == '__main__':
    unittest.main()