"""
Bonus Phase: Hyperparameter Optimization

Objective: Automatically find the best hyperparameters (learning rate and regularization)
for the manual SVM implementation using Cross-Validation.
"""
import numpy as np
from logging import Logger
from typing import Dict, Any, List
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from src.phase_two import RecursiveBinaryClassifier

def tune_recursive_model(X: np.ndarray, y: np.ndarray, logger: Logger) -> Dict[str, float]:
    """
    Performs a Grid Search to find the best learning rate and lambda for the manual model.
    """
    logger.info("\n--- Phase III (Bonus): Hyperparameter Tuning ---")
    logger.info("Objective: Optimizing 'learning_rate' and 'lambda_param' via Grid Search.")
    
    # Define hyperparameter grid
    learning_rates: List[float] = [0.0001, 0.001, 0.005, 0.01]
    lambdas: List[float] = [0.001, 0.01, 0.1]
    
    best_acc: float = 0.0
    best_params: Dict[str, float] = {'learning_rate': 0.001, 'lambda_param': 0.01}
    
    # 3-Fold Cross Validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    total_combinations = len(learning_rates) * len(lambdas)
    current_iter = 0

    for lr in learning_rates:
        for lam in lambdas:
            current_iter += 1
            fold_accuracies = []
            
            # Reduce logging noise during tuning
            temp_logger = logger.getChild(f"tuning_{current_iter}")
            temp_logger.disabled = True 
            
            for train_idx, val_idx in kf.split(X):
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                
                model = RecursiveBinaryClassifier(learning_rate=lr, lambda_param=lam, n_iters=1000)
                model.logger = logger # Use main logger but rely on fit logging 
                # To suppress logs effectively we would need to modify the class, 
                # but let's just accept the logs or simply disable the logger passed if possible.
                # Actually RecursiveBinaryClassifier requires a logger.
                # We will accept the verbose output for transparency or use the disabled logger
                model.logger = temp_logger

                model.fit(X_fold_train, y_fold_train)
                preds = model.predict(X_fold_val)
                fold_accuracies.append(accuracy_score(y_fold_val, preds))
            
            avg_acc = np.mean(fold_accuracies)
            # logger.info(f"Tested lr={lr}, lambda={lam} -> Avg Acc: {avg_acc:.4f}")
            
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_params = {'learning_rate': lr, 'lambda_param': lam}

    logger.info(f"Optimization Complete. Best Params: {best_params} with CV Accuracy: {best_acc:.4f}")
    return best_params
