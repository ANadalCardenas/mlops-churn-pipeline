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
