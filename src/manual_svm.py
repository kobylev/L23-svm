"""
Manual SVM Implementation

Objective: A helper module containing the manual implementation of a 
linear Support Vector Machine using Gradient Descent.
"""
import numpy as np

class ManualSVM:
    """A manual implementation of a linear Support Vector Machine using Gradient Descent."""

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
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

    def predict(self, X):
        """Predict the class label for a given input."""
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
