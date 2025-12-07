"""
Manual SVM Implementation

Objective: A helper module containing the manual implementation of a 
linear Support Vector Machine using Gradient Descent.
"""
import numpy as np
from typing import Optional

class ManualSVM:
    """A manual implementation of a linear Support Vector Machine using Gradient Descent."""

    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iters: int = 1000, 
                 decay_rate: float = 0.01, momentum: float = 0.9, random_state: Optional[int] = None,
                 kernel: str = 'linear', gamma: float = 1.0, n_rff_components: int = 100) -> None:
        self.lr: float = learning_rate
        self.lambda_param: float = lambda_param
        self.n_iters: int = n_iters
        self.decay_rate: float = decay_rate
        self.momentum: float = momentum
        self.random_state: Optional[int] = random_state # Store random_state
        self.kernel: str = kernel
        self.gamma: float = gamma
        self.n_rff_components: int = n_rff_components
        
        self.w: Optional[np.ndarray] = None
        self.b: Optional[float] = None
        
        # RFF parameters
        self.rff_W: Optional[np.ndarray] = None
        self.rff_b_offset: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVM by finding the optimal weights (w) and bias (b)
        using vectorized Batch Gradient Descent with Momentum on the Hinge Loss function.
        Supports 'linear' and 'rbf' (via Random Fourier Features) kernels.
        """
        n_samples, n_features = X.shape
        # Ensure y is in the format {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        if self.random_state is not None:
            np.random.seed(self.random_state) # Seed for reproducibility

        # Apply Kernel Trick (RFF) if requested
        if self.kernel == 'rbf':
            # w ~ N(0, 2*gamma)
            self.rff_W = np.random.normal(loc=0, scale=np.sqrt(2 * self.gamma), size=(n_features, self.n_rff_components))
            # b ~ U(0, 2pi)
            self.rff_b_offset = np.random.uniform(0, 2 * np.pi, self.n_rff_components)
            
            # Transform X -> Z
            X_train = np.sqrt(2 / self.n_rff_components) * np.cos(np.dot(X, self.rff_W) + self.rff_b_offset)
            current_n_features = self.n_rff_components
        else:
            X_train = X
            current_n_features = n_features

        # Initialize weights and bias
        self.w = np.random.randn(current_n_features) * 0.01 # Initialize with small random values
        self.b = 0.0 # Bias initialized to zero
        
        # Initialize velocities for momentum
        velocity_w = np.zeros(current_n_features)
        velocity_b = 0.0
        
        prev_loss = float('inf')
        
        initial_lr_for_epoch = self.lr # Store the initial lr for this fit call

        for epoch in range(self.n_iters):
            # Apply learning rate decay
            current_lr = initial_lr_for_epoch / (1 + self.decay_rate * epoch) 

            # Calculate scores for the entire batch
            scores = np.dot(X_train, self.w) - self.b
            
            # Identify misclassified points or points within the margin (conditions < 1)
            margins = y_ * scores
            misclassified_indicators = np.where(margins < 1, 1, 0)
            
            # Calculate gradients
            dw = (2 * self.lambda_param * self.w) - np.dot(X_train.T, (y_ * misclassified_indicators))
            db = np.sum(y_ * misclassified_indicators)
            
            # Update parameters using Momentum
            velocity_w = self.momentum * velocity_w - current_lr * dw
            velocity_b = self.momentum * velocity_b - current_lr * db
            
            self.w += velocity_w
            self.b += velocity_b
            
            # --- Convergence Check ---
            # Calculate current loss: Average Hinge Loss + Regularization
            hinge_loss = np.maximum(0, 1 - margins)
            loss = np.mean(hinge_loss) + self.lambda_param * np.dot(self.w, self.w)
            
            if abs(prev_loss - loss) < 1e-5:
                # print(f"Converged at epoch {epoch}")
                break
            
            prev_loss = loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class label for a given input."""
        if self.w is None or self.b is None:
             raise RuntimeError("Model has not been fitted yet.")
        
        if self.kernel == 'rbf':
             if self.rff_W is None or self.rff_b_offset is None:
                 raise RuntimeError("RFF parameters not initialized.")
             X_transformed = np.sqrt(2 / self.n_rff_components) * np.cos(np.dot(X, self.rff_W) + self.rff_b_offset)
             approx = np.dot(X_transformed, self.w) - self.b
        else:
             approx = np.dot(X, self.w) - self.b
             
        return np.sign(approx)
