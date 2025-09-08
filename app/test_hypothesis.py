"""
Pytest for hypothesis testing APIs.
We train the model first using the same TestClient instance so endpoints have data.
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def train_once():
    # Train model once per module
    r = client.post("/train_model", json={"test_size": 0.2, "random_state": 1, "n_neighbors": 3})
    assert r.status_code == 200

def test_feature_significance_valid():
    r = client.post("/feature_significance", json={"feature": "alcohol", "bins": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert "p_value" in data

def test_feature_significance_invalid_feature():
    r = client.post("/feature_significance", json={"feature": "no_such_feature", "bins": 3})
    assert r.status_code == 400

def test_feature_correlation_valid():
    r = client.post("/feature_correlation", json={"f1": "alcohol", "f2": "malic_acid", "method": "pearson"})
    assert r.status_code == 200
    data = r.json()
    assert "correlation" in data

def test_feature_normality_valid():
    r = client.post("/feature_normality", json={"feature": "alcohol"})
    assert r.status_code == 200
    data = r.json()
    assert "p_value" in data

def test_model_vs_random():
    r = client.post("/model_vs_random", json={})
    assert r.status_code == 200
    data = r.json()
    assert "accuracy" in data and "p_value" in data

# def test_feature_importance():
#     r = client.post("/feature_importance", json={})
#     assert r.status_code == 200
#     data = r.json()
#     assert "importances_mean" in data

def test_feature_combination_manova():
    # choose two numeric features
    r = client.post("/feature_combination", json={"features": ["alcohol", "malic_acid"]})
    # may return a complex result or status success
    assert r.status_code == 200

# def test_feature_interaction():
#     r = client.post("/feature_interaction", json={"f1": "alcohol", "f2": "malic_acid"})
#     assert r.status_code == 200

# def test_model_consistency():
#     r = client.post("/model_consistency", json={"n_splits": 4})
#     assert r.status_code == 200
#     data = r.json()
#     assert "accuracies" in data
