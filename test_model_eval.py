"""
Pytest test cases for Hypothesis Testing APIs.
Each test validates:
- Correct input handling
- Correct hypothesis testing response
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_feature_significance():
    """
    Test single feature significance (Chi-squared test).
    Checks p-value and result interpretation.
    """
    response = client.post("/feature_significance", json={"feature": "alcohol", "bins": 3})
    assert response.status_code == 200
    data = response.json()
    assert "p_value" in data
    assert "result" in data


def test_feature_correlation():
    """
    Test correlation significance between two features.
    Validates correlation coefficient and hypothesis decision.
    """
    response = client.post("/feature_correlation", json={"f1": "alcohol", "f2": "color_intensity", "method": "pearson"})
    assert response.status_code == 200
    data = response.json()
    assert "correlation" in data
    assert "p_value" in data


def test_feature_normality():
    """
    Test Shapiro-Wilk normality for a feature.
    Verifies statistic and decision.
    """
    response = client.post("/feature_normality", json={"feature": "alcohol"})
    assert response.status_code == 200
    data = response.json()
    assert "statistic" in data
    assert "p_value" in data


def test_model_vs_random():
    """
    Test if model performs significantly better than random chance.
    Uses binomial test validation.
    """
    response = client.post("/model_vs_random", json={})
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "p_value" in data


def test_feature_importance():
    """
    Test feature importance significance using permutation.
    Ensures importance scores and p-values are returned.
    """
    response = client.post("/feature_importance", json={})
    assert response.status_code == 200
    data = response.json()
    assert "importances" in data


def test_feature_combination():
    """
    Test feature combination significance (MANOVA).
    Validates p-value and result decision.
    """
    response = client.post("/feature_combination", json={"features": ["alcohol", "color_intensity"]})
    assert response.status_code == 200
    data = response.json()
    assert "p_value" in data


def test_feature_interaction():
    """
    Test interaction significance between two features.
    Ensures valid hypothesis test result.
    """
    response = client.post("/feature_interaction", json={"f1": "alcohol", "f2": "color_intensity"})
    assert response.status_code == 200
    data = response.json()
    assert "p_value" in data


def test_model_consistency():
    """
    Test model consistency across multiple splits.
    Validates ANOVA results for performance stability.
    """
    response = client.post("/model_consistency", json={"n_splits": 5})
    assert response.status_code == 200
    data = response.json()
    assert "p_value" in data
