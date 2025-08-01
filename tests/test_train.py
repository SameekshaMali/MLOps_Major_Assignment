# tests/test_train.py

import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.train import load_data, train_model

# Load dataset once for all tests
X, y = load_data()

def test_data_loaded():
    assert X is not None and y is not None
    assert X.shape[0] == len(y)
    print("Data is loaded correctly..")

def test_model_instance():
    model = train_model(X, y)
    assert isinstance(model, LinearRegression)
    print("Model is instance of LinearRegression..")

def test_model_trained():
    model = train_model(X, y)
    assert hasattr(model, 'coef_')
    print("Model trained and has coefficients..")

def test_model_performance():
    model = train_model(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    assert r2 > 0.5  # Can adjust based on your result
    print(f"Model R2 Score: {r2:.4f}")
