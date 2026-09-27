"""
Inference entrypoint. Loads the current Production model from the MLflow Model
Registry, predicts on a single input record, and appends the result to a CSV
file that emulates a prediction database.

1. Resolves the `Production` alias of `churn-model` to a concrete registry version
2. Loads exactly that version
3. Generates a prediction for the JSON input record
4. Appends prediction_id, timestamp, model_version, input_data, and prediction
   to the predictions CSV (writing the header only when the file is new)

"""

import argparse
import json
import os
import uuid
from datetime import datetime

import mlflow
import pandas as pd

MODEL_NAME = "churn-model"
MODEL_ALIAS = "Production"


def run_inference(input_data: dict, output_path: str) -> dict:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

    # Resolve the alias first and load by explicit version, so the alias can't move
    # between loading the model and recording which version produced the prediction
    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{model_version}")

    prediction = model.predict(pd.DataFrame([input_data]))[0]

    record = {
        "prediction_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "input_data": json.dumps(input_data),
        "prediction": int(prediction),
    }

    # Append mode keeps every past prediction; the header is only written for a new file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pd.DataFrame([record]).to_csv(
        output_path, mode="a", index=False, header=not os.path.exists(output_path)
    )

    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)  # JSON object with one customer's features
    parser.add_argument("--output", default="predictions/predictions.csv")
    args = parser.parse_args()

    print(json.dumps(run_inference(json.loads(args.input), args.output)))
