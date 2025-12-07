# src/plotting.py
"""
Utility functions for creating and saving plots for the SVM assignment.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Any

def plot_iris_data(X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None) -> None:
    """Plots the Iris dataset features (petal length vs width)."""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Iris Dataset - Petal Features")
    plt.legend(handles=scatter.legend_elements()[0], labels=['Setosa', 'Versicolor', 'Virginica'])
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved data visualization to {save_path}")
    plt.close()

def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model: Any, title: str, save_path: Optional[str] = None) -> None:
    """
    Plots the decision boundary of a trained classifier along with the test data.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # The `RecursiveBinaryClassifier` needs a `predict` method that works on meshgrid
    if hasattr(model, 'predict_on_mesh'):
        Z = model.predict_on_mesh(np.c_[xx.ravel(), yy.ravel()])
    else: # For sklearn models
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved decision boundary plot to {save_path}")
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: Optional[str] = None) -> None:
    """Plots the confusion matrix for the given predictions."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    plt.close()