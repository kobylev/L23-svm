# run.py
"""
Main execution script for the SVM Classification Assignment.

This script orchestrates the following tasks:
1. Sets up logging to capture all output to `logs/execution.log`.
2. Runs Phase I (scikit-learn implementation) and logs its accuracy.
3. Runs Phase II (manual implementation) and logs its accuracy.
4. Generates and saves plots for data visualization and decision boundaries.
"""
import os
import logging
from sklearn import datasets
from sklearn.model_selection import train_test_split

from src.phase_one import run_phase_one
from src.phase_two import run_phase_two, RecursiveBinaryClassifier
from src.tuning import tune_recursive_model
from src.plotting import plot_iris_data, plot_decision_boundary, plot_confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np

def main() -> None:
    """Main function to run the project."""
    # --- 1. Setup Logging and Directories ---
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler("logs/execution.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    # --- 2. Load Data ---
    iris = datasets.load_iris()
    # Use only petal length and petal width for clear 2D visualization
    X = iris.data[:, [2, 3]]
    y = iris.target

    # --- 3. Split Data ---
    # The data split is now done once in the main script for consistency.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # --- 3. Generate Initial Data Plot ---
    plot_iris_data(X, y, save_path="plots/iris_data_visualization.png")

    # --- 4. Run Phase I and Plot ---
    model1 = run_phase_one(X_train, y_train, X_test, y_test, logger)
    plot_decision_boundary(X_test, y_test, model1,
                           "Phase I: Scikit-Learn SVM Decision Boundary",
                           save_path="plots/decision_boundary_phase_one.png")

    # --- 5. Run Phase II and Plot ---
    model2 = run_phase_two(X_train, y_train, X_test, y_test, logger, random_state=42)
    plot_decision_boundary(X_test, y_test, model2,
                           "Phase II: Manual SVM Decision Boundaries",
                           save_path="plots/decision_boundary_phase_two.png")

    # --- 6. Run Phase III (Bonus) ---
    # Retrieve best parameters (using hardcoded for now, or from a loaded optimization result)
    # The tuning process itself uses random_state=42 for CV splits, ensuring reproducibility of the tuning
    best_params = {'learning_rate': 0.0039, 'lambda_param': 0.0005, 'n_iters': 10000, 'decay_rate': 0.01} # Hardcoded best from previous optuna run
    
    # Train final optimized model
    model3 = RecursiveBinaryClassifier(
        learning_rate=best_params['learning_rate'],
        lambda_param=best_params['lambda_param'],
        n_iters=best_params['n_iters'],
        decay_rate=best_params['decay_rate'],
        random_state=42 # Ensure reproducibility of the final model
    )
    model3.logger = logger
    model3.fit(X_train, y_train)
    
    y_pred_opt = model3.predict(X_test)
    acc_opt = accuracy_score(y_test, y_pred_opt)
    
    logger.info(f"Phase III Optimized Accuracy: {acc_opt:.4f}")
    
    plot_decision_boundary(X_test, y_test, model3,
                           "Phase III: Optimized Manual SVM",
                           save_path="plots/decision_boundary_phase_three_optimized.png")
    
    plot_confusion_matrix(y_test, y_pred_opt, 
                          "Confusion Matrix - Optimized Manual SVM",
                          save_path="plots/confusion_matrix_optimized.png")

if __name__ == '__main__':
    main()