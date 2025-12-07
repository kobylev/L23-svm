"""
Manual SVM Implementation

Objective: A helper module containing the manual implementation of a 
linear Support Vector Machine using Gradient Descent.
"""
import numpy as np
from typing import Optional

class ManualSVM:
    """A manual implementation of a linear Support Vector Machine using Gradient Descent."""

    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iters: int = 1000, decay_rate: float = 0.01, random_state: Optional[int] = None) -> None:
        self.lr: float = learning_rate
        self.lambda_param: float = lambda_param
        self.n_iters: int = n_iters
        self.decay_rate: float = decay_rate
        self.random_state: Optional[int] = random_state # Store random_state
        self.w: Optional[np.ndarray] = None
        self.b: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVM by finding the optimal weights (w) and bias (b)
        using vectorized Batch Gradient Descent on the Hinge Loss function.
        """
        n_samples, n_features = X.shape
        # Ensure y is in the format {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        if self.random_state is not None:
            np.random.seed(self.random_state) # Seed for reproducibility
        self.w = np.random.randn(n_features) * 0.01 # Initialize with small random values
        self.b = 0.0 # Bias initialized to zero
        
        prev_loss = float('inf')
        
        initial_lr_for_epoch = self.lr # Store the initial lr for this fit call

        for epoch in range(self.n_iters):
            # Apply learning rate decay
            # Formula is: self.lr = self.lr / (1 + self.decay_rate * epoch)
            # Using initial_lr_for_epoch to avoid cumulative decay affecting subsequent fit calls
            current_lr = initial_lr_for_epoch / (1 + self.decay_rate * epoch) 

            # Calculate scores for the entire batch
            scores = np.dot(X, self.w) - self.b
            
            # Identify misclassified points or points within the margin (conditions < 1)
            margins = y_ * scores
            misclassified_indicators = np.where(margins < 1, 1, 0)
            
            # Calculate gradients
            dw = (2 * self.lambda_param * self.w) - np.dot(X.T, (y_ * misclassified_indicators))
            db = np.sum(y_ * misclassified_indicators)
            
            # Update parameters using the current decayed learning rate
            self.w -= current_lr * dw
            self.b -= current_lr * db
            
            # --- Convergence Check ---
            # Calculate current loss: Average Hinge Loss + Regularization
            hinge_loss = np.maximum(0, 1 - margins)
            loss = np.mean(hinge_loss) + self.lambda_param * np.dot(self.w, self.w)
            
            if abs(prev_loss - loss) < 1e-5:
                print(f"Converged at epoch {epoch}")
                break
            
            prev_loss = loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class label for a given input."""
        if self.w is None or self.b is None:
             raise RuntimeError("Model has not been fitted yet.")
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
