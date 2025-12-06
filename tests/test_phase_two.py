import unittest
import numpy as np
import logging
from src.phase_two import RecursiveBinaryClassifier

class TestPhaseTwo(unittest.TestCase):
    def setUp(self):
        # Create dummy data for testing 3 classes with aggressively wider margins
        # Class 0: negative coordinates
        # Class 1: near origin
        # Class 2: positive coordinates
        self.X_train = np.array([
            [-10.0, -10.0], [-9.0, -9.0], 
            [0.0, 0.0], [1.0, 1.0], 
            [10.0, 10.0], [11.0, 11.0]
        ])
        self.y_train = np.array([0, 0, 1, 1, 2, 2])
        self.X_test = np.array([[-9.5, -9.5], [0.5, 0.5], [10.5, 10.5]])
        self.y_test = np.array([0, 1, 2])
        
        # Mock logger
        self.logger = logging.getLogger('test_logger')
        self.logger.disabled = True

    def test_recursive_classifier_execution(self):
        """Test if RecursiveBinaryClassifier runs without errors."""
        model = RecursiveBinaryClassifier(n_iters=500)
        model.logger = self.logger
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_fitted)

    def test_recursive_classifier_prediction(self):
        """Test if RecursiveBinaryClassifier predicts correctly on simple data."""
        # Increased iterations and learning rate tuning
        model = RecursiveBinaryClassifier(n_iters=5000, learning_rate=0.001)
        model.logger = self.logger
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        np.testing.assert_array_equal(predictions, self.y_test)

    def test_predict_before_fit_raises_error(self):
        """Test that calling predict before fit raises a RuntimeError."""
        model = RecursiveBinaryClassifier()
        with self.assertRaises(RuntimeError):
            model.predict(self.X_test)

if __name__ == '__main__':
    unittest.main()
