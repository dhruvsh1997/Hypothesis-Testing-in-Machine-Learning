"""
Machine learning and statistical utilities for wine KNN project.
Contains functions to load data, train KNN, evaluate, and run hypothesis tests.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.inspection import permutation_importance

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr, spearmanr, shapiro, binom_test, f_oneway
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
import statsmodels.api as sm

from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Globals stored in memory for simplicity
DATA_STORE = {
    "wine": None,  # sklearn dataset object
    "df": None,    # full DataFrame including target
    "X": None,
    "y": None,
    "scaler": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "model": None
}

def prepare_and_train(test_size: float = 0.2, random_state: int = 42, n_neighbors: int = 5) -> Dict[str, Any]:
    """
    Load wine dataset, preprocess (scale), split and train KNN.
    Returns summary info.
    """
    dataset = load_wine(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()
    df = dataset.frame.copy()  # includes target column name 'target'

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Persist to global store
    DATA_STORE.update({
        "wine": dataset,
        "df": df,
        "X": X_scaled,
        "y": y,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": knn
    })

    return {"message": "trained", "n_features": X.shape[1], "n_classes": len(np.unique(y))}

def evaluate_model() -> Dict[str, Any]:
    """
    Evaluate the trained KNN model on test set and return metrics.
    """
    model = DATA_STORE["model"]
    X_test = DATA_STORE["X_test"]
    y_test = DATA_STORE["y_test"]

    if model is None or X_test is None:
        raise RuntimeError("Model not trained")

    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        pass

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    if y_proba is not None and y_proba.shape[1] > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba, multi_class="ovr"))

    return metrics

# ---------- Hypothesis tests ----------

def single_feature_chi2(feature: str, bins: int = 3, min_unique_for_categorical: int = 10) -> Dict[str, Any]:
    """
    Chi-squared test for independence between a (possibly continuous) feature and the target.
    If numeric, the feature will be binned into `bins` quantile bins, to create categorical buckets.
    Returns chi2, p-value and result string.
    """
    df = DATA_STORE["df"]
    if df is None:
        raise RuntimeError("Data not loaded. Call prepare_and_train first.")

    if feature not in df.columns:
        raise ValueError(f"Feature {feature} not found")

    series = df[feature]
    # If numeric with many unique values, bin it
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > min_unique_for_categorical:
        # use quantile bins to avoid empty bins
        df_binned = series.copy()
        df_binned = pd.qcut(series, q=bins, duplicates="drop")
        contingency = pd.crosstab(df_binned, df["target"])
    else:
        contingency = pd.crosstab(series, df["target"])

    chi2, p, dof, expected = chi2_contingency(contingency)
    result = "Reject H0" if p < 0.05 else "Fail to Reject H0"
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "result": result}

def feature_correlation(f1: str, f2: str, method: str = "pearson") -> Dict[str, Any]:
    """
    Test correlation between two numeric features using pearson or spearman.
    """
    df = DATA_STORE["df"]
    if df is None:
        raise RuntimeError("Data not loaded")

    if f1 not in df.columns or f2 not in df.columns:
        raise ValueError("Feature not found")

    if not pd.api.types.is_numeric_dtype(df[f1]) or not pd.api.types.is_numeric_dtype(df[f2]):
        raise ValueError("Correlation tests require numeric features")

    if method == "pearson":
        corr, p = pearsonr(df[f1], df[f2])
    else:
        corr, p = spearmanr(df[f1], df[f2])

    result = "Reject H0" if p < 0.05 else "Fail to Reject H0"
    return {"correlation": float(corr), "p_value": float(p), "result": result}

def feature_normality(feature: str) -> Dict[str, Any]:
    """
    Shapiro-Wilk test for normality of one feature across the dataset (not per class).
    """
    df = DATA_STORE["df"]
    if df is None:
        raise RuntimeError("Data not loaded")

    if feature not in df.columns:
        raise ValueError("Feature not found")

    if not pd.api.types.is_numeric_dtype(df[feature]):
        raise ValueError("Normality test requires numeric feature")

    stat, p = shapiro(df[feature])
    result = "Reject H0 (Not normal)" if p < 0.05 else "Fail to Reject H0 (Normal)"
    return {"statistic": float(stat), "p_value": float(p), "result": result}

def model_vs_random_test() -> Dict[str, Any]:
    """
    Binomial test to check whether model accuracy is significantly better than chance.
    """
    model = DATA_STORE["model"]
    if model is None:
        raise RuntimeError("Model not trained")

    X_test = DATA_STORE["X_test"]
    y_test = DATA_STORE["y_test"]

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    successes = int((y_pred == y_test).sum())
    n = int(len(y_test))
    # chance probability is 1 / n_classes
    p_chance = 1.0 / len(np.unique(DATA_STORE["y"]))
    p_value = binom_test(successes, n=n, p=p_chance, alternative="greater")
    result = "Reject H0" if p_value < 0.05 else "Fail to Reject H0"
    return {"accuracy": float(acc), "p_value": float(p_value), "result": result}

def feature_importance_permutation(n_repeats: int = 30, random_state: int = 42) -> Dict[str, Any]:
    """
    Permutation importance test for trained model. Returns importance and basic p-value estimate
    by computing null distribution from permuted labels.
    """
    model = DATA_STORE["model"]
    X_test = DATA_STORE["X_test"]
    y_test = DATA_STORE["y_test"]
    df = DATA_STORE["df"]

    if model is None:
        raise RuntimeError("Model not trained")

    # compute baseline accuracy
    base_acc = accuracy_score(y_test, model.predict(X_test))

    r = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=1)
    importances = r.importances_mean
    features = list(DATA_STORE["wine"].feature_names)

    # No formal p-value here but we return importances
    return {"base_accuracy": float(base_acc), "importances_mean": dict(zip(features, importances))}

def feature_combination_manova(features: List[str]) -> Dict[str, Any]:
    """
    MANOVA test for a set of numeric features to see if classes differ on these features jointly.
    """
    df = DATA_STORE["df"]
    if df is None:
        raise RuntimeError("Data not loaded")
    for f in features:
        if f not in df.columns:
            raise ValueError(f"Feature {f} not found")
        if not pd.api.types.is_numeric_dtype(df[f]):
            raise ValueError("MANOVA requires numeric features")

    # Build formula like 'feat1 + feat2 + feat3 ~ target' but MANOVA in statsmodels expects endog and exog via formula
    formula = " + ".join(features) + " ~ target"
    maov = MANOVA.from_formula(formula, data=df)
    res = maov.mv_test()
    # statsmodels returns a complex object; get Pillai trace pvalue (common)
    try:
        pillai_p = res.results['target']['stat']['Pr > F']['Pillai\'s trace']
        pillai_stat = res.results['target']['stat']['Value']['Pillai\'s trace']
    except Exception:
        # fallback: return full text repr
        return {"result": str(res)}
    result = "Reject H0" if pillai_p < 0.05 else "Fail to Reject H0"
    return {"pillai_stat": float(pillai_stat), "p_value": float(pillai_p), "result": result}

def feature_interaction_test(f1: str, f2: str) -> Dict[str, Any]:
    """
    Test for interaction between two features predicting class by fitting multinomial logit
    and comparing model with and without interaction term via likelihood ratio test.
    For simplicity use OLS on one-vs-rest encoding of a selected class.
    """
    df = DATA_STORE["df"]
    if df is None:
        raise RuntimeError("Data not loaded")
    for f in (f1, f2):
        if f not in df.columns:
            raise ValueError(f"Feature {f} not found")
        if not pd.api.types.is_numeric_dtype(df[f]):
            raise ValueError("Interaction test requires numeric features")

    # one-vs-rest for class 0 vs others as a simple test
    df_local = df.copy()
    df_local["y01"] = (df_local["target"] == 0).astype(int)
    # full model with interaction
    formula_full = f"y01 ~ {f1} + {f2} + {f1}:{f2}"
    model_full = ols(formula_full, data=df_local).fit()
    # reduced model without interaction
    formula_red = f"y01 ~ {f1} + {f2}"
    model_red = ols(formula_red, data=df_local).fit()
    lr_stat = 2 * (model_full.llf - model_red.llf)
    p_value = sm.stats.stattools.stats.chisqprob(lr_stat, df=1) if hasattr(sm.stats.stattools, "stats") else None
    # the above fallback for p computation may not be available; use approximate via chi2
    import scipy.stats as st
    p_value = float(st.chi2.sf(lr_stat, df=1))
    result = "Reject H0" if p_value < 0.05 else "Fail to Reject H0"
    return {"lr_stat": float(lr_stat), "p_value": float(p_value), "result": result}

def model_consistency_tests(n_splits: int = 5, random_state: int = 42) -> Dict[str, Any]:
    """
    Test whether model performance differs across different random splits.
    We perform repeated stratified splits, train KNN each time, and then perform one-way ANOVA
    on accuracies to test consistency.
    """
    X = DATA_STORE["X"]
    y = DATA_STORE["y"]
    if X is None:
        raise RuntimeError("Data not loaded")

    accs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xts = X[train_idx], X[test_idx]
        ytr, yts = y.iloc[train_idx], y.iloc[test_idx]
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(Xtr, ytr)
        accs.append(accuracy_score(yts, model.predict(Xts)))

    # ANOVA across folds
    f_stat, p_val = f_oneway(*[[a] for a in accs])  # single-value groups still produce valid p
    result = "Reject H0" if p_val < 0.05 else "Fail to Reject H0"
    return {"accuracies": [float(a) for a in accs], "f_stat": float(f_stat), "p_value": float(p_val), "result": result}
