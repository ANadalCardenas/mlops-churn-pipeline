"""
Builds and returns the scikit-learn preprocessing + model Pipeline.

Detects numeric and categorical columns from the input DataFrame, then
constructs a ColumnTransformer with:
  - Numeric branch: SimpleImputer(strategy="median")
  - Categorical branch: SimpleImputer(strategy="most_frequent") + OneHotEncoder

Wraps the transformer in a Pipeline with LogisticRegression(max_iter=1000) as
the final estimator. Does NOT fit the pipeline — fitting happens in train.py.

Public API:
    build_training_pipeline(df: pd.DataFrame) -> sklearn.pipeline.Pipeline
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_training_pipeline(df: pd.DataFrame) -> Pipeline:
    # These columns are not features: customerID is an identifier, Churn is the target
    exclude = {"customerID", "Churn"}

    numeric_cols = [c for c in df.select_dtypes(include=["int64", "float64"]).columns if c not in exclude]
    categorical_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in exclude]

    numeric_transformer = SimpleImputer(strategy="median")

    # Categorical columns need two steps: fill missing values first, then encode
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    # Combine preprocessing and the model into a single callable pipeline
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
