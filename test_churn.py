import pandas as pd
import numpy as np
from churn_prediction import load_data, preprocess_data, train_models, evaluate_models

def test_load_data():
    """Test if data loading works correctly."""
    data = load_data("data/customer_churn.csv")
    assert isinstance(data, pd.DataFrame)
    assert "churn" in data.columns
    assert data.shape[1] == 5  # 4 features + 1 target

def test_preprocess_data():
    """Test if preprocessing returns correct shapes."""
    data = load_data("data/customer_churn.csv")
    X, y = preprocess_data(data)
    assert X.shape[0] == data.shape[0]
    assert X.shape[1] == 5  # 3 numerical + 2 dummy variables for contract_type
    assert len(y) == data.shape[0]

def test_train_models():
    """Test if all models can be trained."""
    data = load_data("customer_churn.csv")
    X, y = preprocess_data(data)
    models_dict, X_train, X_test, y_train, y_test = train_models(X, y)
    assert "Logistic Regression" in models_dict
    assert "Decision Tree" in models_dict
    assert "Random Forest" in models_dict
    assert X_test is not None
    assert y_test is not None

def test_evaluate_models():
    """Test if evaluation metrics are reasonable."""
    data = load_data("customer_churn.csv")
    X, y = preprocess_data(data)
    models_dict, X_train, X_test, y_train, y_test = train_models(X, y)
    results = evaluate_models(models_dict, X_train, X_test, y_train, y_test)
    assert isinstance(results, pd.DataFrame)
    assert "Training Accuracy" in results.columns
    assert "Testing Accuracy" in results.columns
    assert "AUROC Score" in results.columns
    for score in results["AUROC Score"]:
        assert 0 <= float(score) <= 1
