"""
Simple plotting utilities for the project.
Saves images to temporary files and returns file paths.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
import pandas as pd
from .ml_utils import DATA_STORE

def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(conf_matrix))
    disp.plot(ax=ax)
    ax.set_title("Confusion Matrix")
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(f.name, bbox_inches="tight")
    plt.close(fig)
    return f.name

def plot_roc(y_test, y_proba):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_proba.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # aggregate micro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["micro"] = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(all_fpr, mean_tpr, label=f"micro-average ROC (AUC = {roc_auc['micro']:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(f.name, bbox_inches="tight")
    plt.close(fig)
    return f.name
