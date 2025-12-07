"""
Manual SVM Implementation

Objective: A helper module containing the manual implementation of a 
linear Support Vector Machine using Gradient Descent.
"""
import numpy as np
from typing import Optional

class ManualSVM:
    """A manual implementation of a linear Support Vector Machine using Gradient Descent."""

    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iters: int = 1000) -> None:
        self.lr: float = learning_rate
        self.lambda_param: float = lambda_param
        self.n_iters: int = n_iters
        self.w: Optional[np.ndarray] = None
        self.b: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVM by finding the optimal weights (w) and bias (b)
        using gradient descent on the Hinge Loss function.
        """
        n_samples, n_features = X.shape
        # Ensure y is in the format {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # Gradient of the regularization term
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # Gradient of the loss term + regularization term
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]
                
                # Update rule
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class label for a given input."""
        if self.w is None or self.b is None:
             raise RuntimeError("Model has not been fitted yet.")
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
