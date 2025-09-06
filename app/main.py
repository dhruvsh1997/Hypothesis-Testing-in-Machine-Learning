"""
FastAPI application exposing endpoints for:
- Training and evaluating a KNN classifier on the Wine dataset
- Running multiple hypothesis testing scenarios to validate statistical significance

All hypothesis endpoints are POST requests and inputs are validated via Pydantic models.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint
from typing import List, Optional, Literal
from .ml_utils import (
    prepare_and_train,
    evaluate_model,
    single_feature_chi2,
    feature_correlation,
    feature_normality,
    model_vs_random_test,
    feature_importance_permutation,
    feature_combination_manova,
    feature_interaction_test,
    model_consistency_tests,
)

# Create FastAPI instance
app = FastAPI(title="Wine KNN Hypothesis API")


# ----------------------------
# Request Models using Pydantic
# ----------------------------
class TrainRequest(BaseModel):
    """Request body for training the KNN model."""
    test_size: float = Field(0.2, ge=0.05, le=0.5)  # Fraction of data used for testing
    random_state: int = Field(42, ge=0)  # Random seed for reproducibility
    n_neighbors: conint(ge=1, le=30) = 5  # K value for KNN classifier


class EvaluateRequest(BaseModel):
    """Placeholder request body for evaluation (future extensibility)."""
    pass


class SingleFeatureRequest(BaseModel):
    """Request body for single feature significance test (Chi-squared)."""
    feature: str = Field(..., description="Feature name to test")
    bins: Optional[int] = Field(3, ge=2, le=10)  # Number of bins for discretization


class CorrelationRequest(BaseModel):
    """Request body for correlation significance test."""
    f1: str
    f2: str
    method: Literal["pearson", "spearman"] = "pearson"


class NormalityRequest(BaseModel):
    """Request body for Shapiro-Wilk normality test."""
    feature: str


class FeatureComboRequest(BaseModel):
    """Request body for MANOVA test with multiple features."""
    features: List[str] = Field(..., min_items=2, max_items=10)


class InteractionRequest(BaseModel):
    """Request body for interaction test between two features."""
    f1: str
    f2: str


class ConsistencyRequest(BaseModel):
    """Request body for model consistency analysis using cross-validation."""
    n_splits: int = Field(5, ge=2, le=20)


# ----------------------------
# Core APIs
# ----------------------------
@app.post("/train_model")
def train_model(req: TrainRequest):
    """
    Train a KNN classifier on the Wine dataset.
    - Splits the dataset
    - Normalizes features
    - Trains the model
    - Returns training/test size and neighbors
    """
    try:
        res = prepare_and_train(
            test_size=req.test_size,
            random_state=req.random_state,
            n_neighbors=req.n_neighbors,
        )
        return {"status": "success", **res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate_model")
def evaluate_model_endpoint(req: EvaluateRequest):
    """
    Evaluate the trained KNN classifier.
    - Computes accuracy, precision, recall, F1-score
    - Generates confusion matrix and ROC-AUC
    """
    try:
        metrics = evaluate_model()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Hypothesis Testing APIs
# ----------------------------
@app.post("/feature_significance")
def api_feature_significance(req: SingleFeatureRequest):
    """
    Test if a single feature significantly affects wine classification.
    Uses Chi-squared test for independence.
    """
    try:
        res = single_feature_chi2(feature=req.feature, bins=req.bins)
        return {"status": "success", **res}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feature_correlation")
def api_feature_correlation(req: CorrelationRequest):
    """
    Test if correlation between two features is statistically significant.
    Supports Pearson (linear correlation) or Spearman (rank correlation).
    """
    try:
        res = feature_correlation(req.f1, req.f2, method=req.method)
        return {"status": "success", **res}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feature_normality")
def api_feature_normality(req: NormalityRequest):
    """
    Test whether a feature is normally distributed across samples.
    Uses Shapiro-Wilk test.
    """
    try:
        res = feature_normality(req.feature)
        return {"status": "success", **res}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model_vs_random")
def api_model_vs_random():
    """
    Test whether the trained model performs significantly better than random guessing.
    Uses Binomial test for validation.
    """
    try:
        res = model_vs_random_test()
        return {"status": "success", **res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feature_importance")
def api_feature_importance():
    """
    Test statistical significance of feature importance using permutation test.
    Identifies which features contribute the most to classification.
    """
    try:
        res = feature_importance_permutation()
        return {"status": "success", **res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feature_combination")
def api_feature_combination(req: FeatureComboRequest):
    """
    Test whether a combination of features significantly affects classification.
    Uses MANOVA for multivariate significance testing.
    """
    try:
        res = feature_combination_manova(req.features)
        return {"status": "success", **res}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feature_interaction")
def api_feature_interaction(req: InteractionRequest):
    """
    Test whether the interaction between two features significantly affects classification.
    Example: Does Alcohol × Color Intensity interaction influence wine type?
    """
    try:
        res = feature_interaction_test(req.f1, req.f2)
        return {"status": "success", **res}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model_consistency")
def api_model_consistency(req: ConsistencyRequest):
    """
    Test whether the model's performance is consistent across different test splits.
    Uses repeated cross-validation and ANOVA test.
    """
    try:
        res = model_consistency_tests(n_splits=req.n_splits)
        return {"status": "success", **res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
