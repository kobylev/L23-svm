import unittest
import numpy as np
import logging
from src.phase_one import run_phase_one

class TestPhaseOne(unittest.TestCase):
    def setUp(self):
        # Create dummy data for testing
        # 3 classes: 0, 1, 2
        # Simple linearly separable data
        self.X_train = np.array([[1, 1], [2, 2], [5, 5], [6, 6], [9, 9], [10, 10]])
        self.y_train = np.array([0, 0, 1, 1, 2, 2])
        self.X_test = np.array([[1.5, 1.5], [5.5, 5.5], [9.5, 9.5]])
        self.y_test = np.array([0, 1, 2])
        
        # Mock logger
        self.logger = logging.getLogger('test_logger')
        self.logger.disabled = True

    def test_phase_one_execution(self):
        """Test if phase one runs without errors and returns a model."""
        model = run_phase_one(self.X_train, self.y_train, self.X_test, self.y_test, self.logger)
        self.assertIsNotNone(model)
        
    def test_phase_one_accuracy(self):
        """Test if phase one model predicts correctly on simple data."""
        model = run_phase_one(self.X_train, self.y_train, self.X_test, self.y_test, self.logger)
        predictions = model.predict(self.X_test)
        np.testing.assert_array_equal(predictions, self.y_test)

if __name__ == '__main__':
    unittest.main()
