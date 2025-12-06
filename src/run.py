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
from src.phase_two import run_phase_two
from src.plotting import plot_iris_data, plot_decision_boundary

def main():
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
    model2 = run_phase_two(X_train, y_train, X_test, y_test, logger)
    plot_decision_boundary(X_test, y_test, model2,
                           "Phase II: Manual SVM Decision Boundaries",
                           save_path="plots/decision_boundary_phase_two.png")

if __name__ == '__main__':
    main()