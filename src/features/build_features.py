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
