"""
Responsible for reading the versioned CSV dataset from disk and returning a
clean pandas DataFrame. Maps data version strings (e.g. "v1", "v2") to their
corresponding file paths, reads the CSV with pandas, and converts the target
column `Churn` from "Yes"/"No" strings to 1/0 integers.

Public API:
    load_dataset(data_version: str) -> pd.DataFrame
"""

import pandas as pd


# Map version strings to their CSV paths; extend here when new data versions arrive
DATA_PATHS = {
    "v1": "data/v1/churn.csv",
    "v2": "data/v2/churn.csv",
}


def load_dataset(data_version: str) -> pd.DataFrame:
    path = DATA_PATHS[data_version]
    df = pd.read_csv(path)

    # Convert the target column from categorical strings to binary integers
    # so sklearn can use it directly as y without extra encoding
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df
