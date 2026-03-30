"""
Main training module. Orchestrates a single end-to-end training run:

1. Loads the dataset for the specified data version via load_data.py
2. Splits features and target (drops customerID and Churn from X)
3. Performs a stratified train/test split (80/20, random_state=42)
4. Builds and fits the sklearn Pipeline from build_features.py
5. Evaluates the model and saves train_metrics.json via evaluate.py and io.py
6. Generates evaluation PNG plots (confusion matrix, ROC, precision-recall)
7. Logs params, metrics, artifacts, and the model artifact to MLflow
8. Saves the model locally under models/<data_version>/model_latest/
9. Writes model_info.json (run ID, git SHA, model URI, timestamp)

Exposes a callable `run_training(...)` function (used by orchestration.py)
and a CLI entry point via argparse for direct invocation.

CLI args: --data-version, --output-dir, --experiment-name, --run-name
"""
