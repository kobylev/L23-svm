"""
Hyperparameter Optimization using Optuna

Objective: Find the optimal hyperparameters (learning_rate, lambda_param, n_iters)
for the ManualSVM implementation to maximize accuracy on the Iris dataset.
"""
import optuna
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import logging

from src.phase_two import RecursiveBinaryClassifier

# Configure logging for Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    """
    Objective function for Optuna optimization.
    """
    # Define hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    lambda_param = trial.suggest_float("lambda_param", 1e-4, 1e-1, log=True)
    n_iters = trial.suggest_int("n_iters", 1000, 10000, step=1000)
    decay_rate = trial.suggest_float("decay_rate", 1e-3, 1e-1, log=True)

    # Load and prepare data (repeated here to ensure clean state for each trial)
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]] # Petal length and width
    y = iris.target
    
            # K-Fold Cross Validation to ensure robustness
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
    
        for train_index, val_index in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]
    
            # Initialize model with suggested parameters and random_state
            model = RecursiveBinaryClassifier(
                learning_rate=learning_rate,
                lambda_param=lambda_param,
                n_iters=n_iters,
                decay_rate=decay_rate,
                random_state=42 # Ensure reproducibility across folds and trials
            )
                    
        # Suppress internal logger during optimization
        model.logger = logging.getLogger("optuna_dummy")
        model.logger.disabled = True

        try:
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_val_fold)
            acc = accuracy_score(y_val_fold, preds)
            accuracies.append(acc)
        except Exception:
            return 0.0 # Return 0 accuracy if fit fails (e.g. divergence)

    return np.mean(accuracies)

def run_optimization():
    print("Starting Optuna Hyperparameter Optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("\n--- Optimization Results ---")
    print(f"Best Trial Accuracy: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params

if __name__ == "__main__":
    run_optimization()
