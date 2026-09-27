import json

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from src.features.build_features import build_training_pipeline
from src.inference import predict


def _register_version(df):
    X = df.drop(columns=["customerID", "Churn"])
    pipeline = build_training_pipeline(X).fit(X, df["Churn"])

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=infer_signature(X, pipeline.predict(X)))

    return mlflow.register_model(f"runs:/{run.info.run_id}/model", predict.MODEL_NAME).version


def test_run_inference_uses_production_version_and_appends_to_csv(tmp_path, monkeypatch, churn_dataframe):
    # Point MLflow at a local file store so the test needs no remote registry
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")

    # Two versions exist, but only the first one carries the Production alias
    production_version = _register_version(churn_dataframe)
    _register_version(churn_dataframe)
    mlflow.MlflowClient().set_registered_model_alias(predict.MODEL_NAME, predict.MODEL_ALIAS, production_version)

    input_data = churn_dataframe.drop(columns=["customerID", "Churn"]).iloc[0].to_dict()
    output_path = tmp_path / "predictions" / "predictions.csv"

    first = predict.run_inference(input_data, str(output_path))
    second = predict.run_inference(input_data, str(output_path))

    rows = pd.read_csv(output_path, dtype={"model_version": str})
    assert list(rows.columns) == ["prediction_id", "timestamp", "model_version", "input_data", "prediction"]
    assert len(rows) == 2
    assert list(rows["prediction_id"]) == [first["prediction_id"], second["prediction_id"]]
    assert set(rows["model_version"]) == {str(production_version)}
    assert json.loads(rows["input_data"][0]) == input_data
    assert set(rows["prediction"]) <= {0, 1}
