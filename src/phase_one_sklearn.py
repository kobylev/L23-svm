# c:\Ai_Expert\L23-svm\phase_one_sklearn.py
"""
Phase I: Standard Library Implementation

Objective: Establish a performance baseline using industry-standard tools.
As per the PRD for the Support Vector Machine (SVM) Classification Assignment.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_phase_one():
    """
    Implements the standard library SVM classification for the Iris dataset.
    """
    # 1. Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 2. Train an SVM model using sklearn.svm
    # Using a linear kernel for simplicity, can be changed to 'rbf', etc.
    svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_classifier.predict(X_test)

    # 3. Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("--- Phase I: Scikit-Learn SVM Implementation ---")
    print(f"Dataset: Iris Flower Dataset")
    print(f"P0 Deliverable: A working sklearn script with high accuracy.")
    print(f"Accuracy on the 3-class data: {accuracy:.4f}")

if __name__ == '__main__':
    run_phase_one()