"""
Computes classification metrics and generates evaluation plots for a fitted
sklearn model.

`evaluate_model` returns a dict with keys: accuracy, precision, recall, f1,
roc_auc — computed via sklearn.metrics on the held-out test set.

`generate_evaluation_plots` saves three PNG files to the specified output
directory:
  - confusion_matrix.png  (ConfusionMatrixDisplay.from_estimator)
  - roc_curve.png         (RocCurveDisplay.from_estimator)
  - precision_recall_curve.png (PrecisionRecallDisplay.from_estimator)

Public API:
    evaluate_model(model, X_test, y_test) -> dict
    generate_evaluation_plots(model, X_test, y_test, output_dir: str) -> None
"""

import os

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of the positive class (Churn=1)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall": recall_score(y_test, y_pred, average="binary"),
        "f1": f1_score(y_test, y_pred, average="binary"),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def generate_evaluation_plots(model, X_test, y_test, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Confusion matrix — shows absolute counts of TP/TN/FP/FN
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # ROC curve — visualises the trade-off between TPR and FPR across thresholds
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    # Precision-recall curve — more informative than ROC when classes are imbalanced
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.close()
