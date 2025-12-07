
import numpy as np
from src.manual_svm import ManualSVM
import pytest

def test_rff_initialization():
    svm = ManualSVM(kernel='rbf', n_rff_components=50, random_state=42)
    X = np.random.randn(10, 5)
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    svm.fit(X, y)
    
    assert svm.rff_W is not None
    assert svm.rff_b_offset is not None
    assert svm.rff_W.shape == (5, 50)
    assert svm.rff_b_offset.shape == (50,)
    assert svm.w.shape == (50,)

def test_rff_prediction():
    svm = ManualSVM(kernel='rbf', n_rff_components=50, random_state=42)
    X = np.random.randn(20, 5)
    y = np.concatenate([np.ones(10), np.zeros(10)])
    
    svm.fit(X, y)
    preds = svm.predict(X)
    assert preds.shape == (20,)
    assert np.all(np.isin(preds, [-1, 1]))
