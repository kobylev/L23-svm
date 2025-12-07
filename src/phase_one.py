"""
Phase I: Standard Library Implementation

Objective: Establish a performance baseline using industry-standard tools.
As per the PRD for the Support Vector Machine (SVM) Classification Assignment.
"""
import numpy as np
from logging import Logger
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_phase_one(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, logger: Logger) -> SVC:
    """
    Implements the standard library SVM classification for the Iris dataset.
    
    Args:
        X_train (np.array): Training feature data.
        y_train (np.array): Training target labels.
        X_test (np.array): Testing feature data.
        y_test (np.array): Testing target labels.
        logger: Logger object for output.
        
    Returns:
        SVC: The trained scikit-learn model.
    """
    # Train an SVM model using a linear kernel
    svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logger.info("\n--- Phase I: Scikit-Learn SVM Implementation ---")
    logger.info(f"Dataset: Iris Flower Dataset")
    logger.info(f"P0 Deliverable: A working sklearn script with high accuracy.")
    logger.info(f"Accuracy on the 3-class data: {accuracy:.4f}")
    
    return svm_classifier