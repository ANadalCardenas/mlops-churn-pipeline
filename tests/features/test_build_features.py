import numpy as np
from sklearn.pipeline import Pipeline

from src.features.build_features import build_training_pipeline


def test_pipeline_structure_excludes_id_and_target_columns(churn_dataframe):
    pipeline = build_training_pipeline(churn_dataframe)

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "classifier"]

    preprocessor = pipeline.named_steps["preprocessor"]
    used_columns = {col for _, _, cols in preprocessor.transformers for col in cols}
    assert "customerID" not in used_columns
    assert "Churn" not in used_columns


def test_pipeline_fits_and_predicts_with_missing_values(churn_dataframe):
    df = churn_dataframe.copy()
    df.loc[0, "tenure"] = np.nan
    df.loc[1, "Contract"] = np.nan

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]

    pipeline = build_training_pipeline(X)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(df)
    assert set(preds).issubset({0, 1})
