import unittest
import numpy as np
from src.manual_svm import ManualSVM

class TestManualSVM(unittest.TestCase):
    def setUp(self):
        # Create dummy binary data for testing (-1 and 1)
        # Class 1: positive coordinates, Class -1: negative coordinates
        self.X = np.array([[2, 2], [3, 3], [-2, -2], [-3, -3]])
        self.y = np.array([1, 1, -1, -1])
        
    def test_fit_predict(self):
        """Test if ManualSVM can fit and predict simple linearly separable data."""
        svm = ManualSVM(n_iters=1000, learning_rate=0.01)
        svm.fit(self.X, self.y)
        
        predictions = svm.predict(self.X)
        np.testing.assert_array_equal(predictions, self.y)

    def test_initialization(self):
        """Test default initialization."""
        svm = ManualSVM()
        self.assertEqual(svm.lr, 0.001)
        self.assertEqual(svm.n_iters, 1000)
        self.assertIsNone(svm.w)
        self.assertIsNone(svm.b)

if __name__ == '__main__':
    unittest.main()
